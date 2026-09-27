"""Binary-backed W8A16/ACC40 Encoder and V1 RVQ. No torch dependency.

The checkpoint supplies only the Encoder graph; all convolution arithmetic
parameters below come from the exported blobs. Streaming holds bounded history.
"""
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

from tools.integer_rvq_reference import clip16, requant, run_integer


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def wide_requant(x, m, s):
    """Python-int fallback: ACC40 times M32 may exceed signed INT64."""
    x = np.asarray(x, dtype=np.int64)
    m, s = int(m), int(s)
    if not 0 <= m <= 2**31-1 or not 0 <= s <= 63:
        raise ValueError('Invalid requant parameters')
    if m == 0:
        return np.zeros_like(x)
    if np.max(np.abs(x), initial=0) <= (2**63-1)//m:
        return requant(x, m, s)
    def scalar(v):
        p = int(v)*m
        a = (abs(p) + ((1 << (s-1)) if s else 0)) >> s
        # Downstream saturation only needs a bounded representative.
        return max(-2**62, min(2**62, -a if p < 0 else a))
    return np.array([scalar(v) for v in x.flat], dtype=np.int64).reshape(x.shape)


def unpack_weight(blob, cout, cin, kernel):
    kg = (cin*kernel+3)//4
    packed = np.frombuffer(blob, dtype='i1').reshape((cout+7)//8, kg, 4, 8)
    w = np.empty((cout, cin, kernel), dtype=np.int64)
    for o in range(cout):
        for k in range(kernel):
            for c in range(cin):
                j = k*cin+c
                w[o,c,k] = packed[o//8,j//4,j%4,o%8]
    return w


class BinaryEncoder:
    def __init__(self, directory, graph):
        root = Path(directory)
        self.manifest = json.loads((root/'encoder_manifest.json').read_text(encoding='utf-8'))
        m = self.manifest
        if (m['weight_bits'],m['activation_bits'],m['accumulator_bits']) != (8,16,40):
            raise ValueError('Only W8A16/ACC40 supported')
        if m['rounding'] != 'half_away_from_zero' or m['zero_point'] != 0:
            raise ValueError('Unsupported Encoder numerical contract')
        wb = (root/'encoder_weights_int8.bin').read_bytes()
        bb = (root/m['bias_file']).read_bytes()
        pb = (root/'encoder_quant_params.bin').read_bytes()
        self.layers = {}
        for a in m['layers']:
            a = dict(a)
            co, ci, k = a['cout'], a['weight_cin_per_group'], a['kernel']
            pc = ((co+7)//8)*8
            off = a['weight_offset']
            a['w'] = unpack_weight(wb[off:off+a['weight_bytes']],co,ci,k)
            a['b'] = np.frombuffer(bb,dtype='<i8',count=pc,offset=a['bias_offset'])[:co]
            off = a['parameter_offset']
            a['m'] = np.frombuffer(pb,dtype='<i4',count=pc,offset=off)[:co]
            a['s'] = np.frombuffer(pb,dtype='u1',count=pc,offset=off+4*pc)[:co]
            if any(pb[off+5*pc:off+6*pc]):
                raise ValueError('Nonzero activation zero point')
            if a['left_pad'] != a['dilation']*(k-1)+1-a['stride'] or a['left_pad'] < 0:
                raise ValueError('Unsupported causal padding')
            self.layers[a['name']] = a
        self.residuals = {r['name']:r for r in m['residuals']}
        if any('alignment' not in r for r in self.residuals.values()):
            raise ValueError('Encoder manifest lacks residual alignment; re-export Encoder first')
        self.graph = graph
        self.reset()

    def reset(self):
        self.history = {}
        self.saturation = {}

    def sat(self, x, name):
        self.saturation[name] = self.saturation.get(name,0)+int(np.count_nonzero((x < -32768)|(x > 32767)))
        return clip16(x)

    def conv(self, name, x):
        a = self.layers[name]
        co, ci, k, stride = a['cout'],a['cin'],a['kernel'],a['stride']
        if x.shape[0] != ci or x.shape[-1] % stride:
            raise ValueError(f'{name}: channel/stride mismatch; use whole 320-sample frames')
        p = a['left_pad']
        old = self.history.get(name,np.zeros((ci,p),dtype=np.int16))
        joined = np.concatenate((old,x),axis=-1)
        self.history[name] = joined[:,-p:].copy() if p else joined[:,:0].copy()
        n = x.shape[-1]//stride
        acc = np.zeros((co,n),dtype=np.int64)
        partial = np.zeros_like(acc)
        wci = a['weight_cin_per_group']
        group = np.arange(co)//(co//a['groups'])
        for j in range(k*wci):
            ki, c = divmod(j,wci)
            pos = np.arange(n)*stride+ki*a['dilation']
            values = joined[group*wci+c][:,pos].astype(np.int64)
            partial += values*a['w'][:,c,ki,None]
            if j%4 == 3 or j == k*wci-1:
                if np.any((partial < -2**39)|(partial > 2**39-1)):
                    raise OverflowError(f'{name}: ACC40 partial overflow')
                acc += partial
                partial.fill(0)
                if np.any((acc < -2**39)|(acc > 2**39-1)):
                    raise OverflowError(f'{name}: ACC40 sum overflow')
        biased = acc+a['b'][:,None]
        self.saturation[name+'/bias40'] = self.saturation.get(name+'/bias40',0)+int(np.count_nonzero((biased < -2**39)|(biased > 2**39-1)))
        biased = np.clip(biased,-2**39,2**39-1)
        raw = np.stack([wide_requant(v,m,s) for v,m,s in zip(biased,a['m'],a['s'])])
        return self.sat(raw,name)

    def run(self, pcm):
        first = self.manifest['layers'][0]
        scale = first['input_scale']
        value = np.asarray(pcm,dtype=np.float64)/32768/scale
        x = self.sat(np.sign(value)*np.floor(np.abs(value)+.5),'input').reshape(1,-1)
        trace = {}
        def walk(node,x,scale):
            kind,name = node['kind'],node.get('name','')
            if kind == 'seq':
                for child in node['children']:
                    x,scale = walk(child,x,scale)
                return x,scale
            if kind == 'conv':
                a = self.layers[name]
                if not math.isclose(scale,a['input_scale'],rel_tol=1e-6):
                    raise ValueError(f'{name}: missing edge scale conversion')
                x = self.conv(name,x)
                trace[name] = x.copy()
                return x,a['output_scale']
            if kind == 'relu':
                if not math.isclose(scale,node['scale'],rel_tol=1e-6):
                    raise ValueError(f'{name}: non-shared ReLU scale unsupported; export explicit activation parameters')
                return np.maximum(x,0),scale
            if kind == 'identity':
                return x,scale
            if kind == 'residual':
                r = self.residuals[name]
                branch,bs = walk(node['branch'],x,scale)
                if not math.isclose(scale,r['identity_scale'],rel_tol=1e-6) or not math.isclose(bs,r['branch_scale'],rel_tol=1e-6):
                    raise ValueError(f'{name}: residual scale mismatch')
                def align(v,key):
                    par = r['alignment'][key]
                    return wide_requant(v,par['multiplier'],par['shift'])
                identity = self.sat(align(x,'identity'),name+'/identity')
                branch = self.sat(align(branch,'branch'),name+'/branch')
                out = self.sat(identity.astype(np.int64)+align(branch,'gain'),name+'/add')
                trace[name+'/add'] = out.copy()
                return out,r['output_scale']
            raise ValueError(kind)
        out,scale = walk(self.graph,x,scale)
        return out,scale,trace


class BinaryRVQ:
    def __init__(self,directory):
        root = Path(directory)
        self.manifest = json.loads((root/'rvq_manifest.json').read_text(encoding='utf-8'))
        m = self.manifest
        if m['format_version'] != 1 or m['rounding'] != 'half_away_from_zero' or m['zero_point'] != 0:
            raise ValueError('Only symmetric V1 RVQ supported')
        for name,sha in m['sha256'].items():
            if Path(name).name != name or digest(root/name) != sha:
                raise ValueError(f'RVQ checksum mismatch: {name}')
        raw = (root/'rvq_codebook_int8.bin').read_bytes()
        norms = (root/'rvq_codebook_norm_int32.bin').read_bytes()
        params = (root/'rvq_stage_requant_params.bin').read_bytes()
        packed_blob = (root/'rvq_codebook_packed_4x8_int8.bin').read_bytes()
        self.books,self.norms,self.params = [],[],[]
        for a in m['stages']:
            k,d = a['codebook_size'],a['codebook_dim']
            b = np.frombuffer(raw,dtype='i1',count=k*d,offset=a['codebook_offset']).reshape(k,d).copy()
            offset = a['packed_offset']
            packed_size = ((k+7)//8)*((d+3)//4)*32
            unpacked = unpack_weight(packed_blob[offset:offset+packed_size],k,d,1)[:,:,0]
            np.testing.assert_array_equal(b,unpacked)
            norm = np.frombuffer(norms,dtype='<i4',count=k,offset=a['norm_offset'])
            np.testing.assert_array_equal(norm,(b.astype(np.int64)**2).sum(-1))
            par = struct.unpack_from('<iBiB',params,a['requant_offset'])
            if par != (a['next_multiplier'],a['next_shift'],a['output_multiplier'],a['output_shift']):
                raise ValueError('RVQ manifest/binary requant mismatch')
            self.books.append(b)
            self.norms.append(norm)
            self.params.append(par)

    def decode(self,indices):
        total = np.zeros((*indices.shape[:-1],self.books[0].shape[1]),dtype=np.int64)
        for q,(b,par) in enumerate(zip(self.books,self.params)):
            if np.any((indices[...,q] < 0)|(indices[...,q] >= len(b))):
                raise ValueError('Invalid code index')
            total += wide_requant(b[indices[...,q]],*par[2:])
        if np.any((total < -2**31)|(total > 2**31-1)):
            raise OverflowError('RVQ output INT32 overflow')
        return clip16(total),int(np.count_nonzero((total < -32768)|(total > 32767)))

    def encode(self,query):
        if np.asarray(query).dtype != np.int16 or query.shape[-1] != self.books[0].shape[1]:
            raise ValueError('RVQ requires INT16 query with exported lookup dimension')
        residual = query.astype(np.int64)
        indices,saturation = [],[]
        for q,(b,norm,par) in enumerate(zip(self.books,self.norms,self.params)):
            score = norm-2*(residual@b.astype(np.int64).T)
            if np.any((score < -2**31)|(score > 2**31-1)):
                raise OverflowError('RVQ distance INT32 overflow')
            index = score.argmin(-1)
            indices.append(index)
            residual -= b[index].astype(np.int64)
            if np.any((residual < -65536)|(residual > 65535)):
                raise OverflowError('RVQ subtraction INT17 overflow')
            raw = wide_requant(residual,*par[:2]) if q+1 < len(self.books) else residual
            saturation.append(int(np.count_nonzero((raw < -32768)|(raw > 32767))) if q+1 < len(self.books) else 0)
            residual = clip16(raw).astype(np.int64)
        indices = np.stack(indices,axis=-1)
        out,sat = self.decode(indices)
        # Independent squared-distance oracle with recomputed parameters.
        oracle,oi,_ = run_integer(query,self.books,[a['scale'] for a in self.manifest['stages']],self.manifest['output_scale'],squared_distance=True)
        np.testing.assert_array_equal(indices,oi)
        np.testing.assert_array_equal(out,oracle)
        return out,indices,dict(residual=saturation,output=sat)
