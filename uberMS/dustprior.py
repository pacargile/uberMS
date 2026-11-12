"""
3D dust map prior on Av, coupled to distance.

The prior p(Av | dist) is tabulated along the line of sight (l,b) before
the fit by reading the 3D dust maps distributed with the dustmaps package,
and linearly interpolated in distance during sampling:

    priors['Av'] = ['3D', {'l':l, 'b':b}]

Where the 3D maps are reliable the prior is a Normal(mu(d), sigma(d))
truncated to [Av_ll, Av_ul]. Closer than the first reliable distance d0 the
prior is flat from 0 up to Av(d0), with a Gaussian fall-off of width
sigma(d0) above it. Beyond the 3D maps, Av ramps to the total extinction
from a 2D map (SFD).

Optional keywords (see build_av3d_table for the full list):

    maps        : list of 3D maps to use, or 'auto' (default) for every
                  map whose footprint covers (l,b)
    source      : 'auto' (default) reads a map from the local dustmaps data
                  directory if it is installed, and otherwise reads only the
                  line of sight from the online archive of the map;
                  'local' or 'remote' force one or the other
    map2d       : 2D total-extinction map used beyond the 3D maps ('sfd')
    sigma       : absolute uncertainty floor in mag (0.05)
    fracsigma   : fractional uncertainty on the map Av (0.1)
    Av_ll,Av_ul : support of the Av prior

A table built with build_av3d_table can also be passed directly, e.g., to
reuse it for several stars along the same line of sight or to save it:

    tab = build_av3d_table(l,b)
    priors['Av'] = ['3D', tab]

Unit conversions to A_V follow the dustmaps documentation
(https://dustmaps.readthedocs.io), see AV_CONVERSION.
"""

import os
import re
import time
import numpy as np

# A_V per unit of each map's native reddening/extinction unit, assuming
# Rv ~ 3.1 (3.32 for DECaPS, as recommended for that map)
AV_CONVERSION = ({
    # Bayestar19 arbitrary units: E(g-r) = 0.901 B19 (Green+2019 eq. 30),
    # so B19 is in SFD E(B-V) units and A_V = 2.742 B19 (Schlafly &
    # Finkbeiner 2011 Table 6, Landolt V, Rv=3.1). The dustmaps docs route
    # E(B-V) = 0.883 B19, A_V = 3.1 E(B-V) = 2.737 B19 agrees to 0.2%.
    'bayestar':      2.742,
    # DECaPS (Zucker+2025) E(B-V) in mag, A_V = 3.32 E(B-V) (dustmaps docs)
    'decaps':        3.32,
    # Edenhofer+2023 E in units of Zhang, Green & Rix (2023),
    # A_V = 2.8 E (Edenhofer et al.)
    'edenhofer2023': 2.8,
    # SFD E(B-V), A_V = 2.742 E(B-V)_SFD (Schlafly & Finkbeiner 2011 Table 6)
    'sfd':           2.742,
    # CSFD (Chiang 2023) is a corrected SFD in the same units
    'csfd':          2.742,
    })

MAPS3D = ['bayestar','decaps','edenhofer2023']
MAPS2D = ['sfd','csfd']

# where each 3D map is found, locally (relative to the dustmaps data
# directory, in order of preference) and online (the archive the dustmaps
# fetch functions download from)
MAPFILES = ({
    'bayestar': {
        'local':['bayestar/bayestar2019.h5'],
        'dataverse':('10.7910/DVN/2EJ9TX','bayestar2019.h5')},
    'decaps': {
        'local':['decaps/decaps_mean_and_samples.h5','decaps/decaps_mean.h5'],
        'dataverse':('10.7910/DVN/J9JCKO','decaps_mean.h5')},
    'edenhofer2023': {
        'local':['edenhofer_2023/mean_and_std_healpix.fits'],
        'url':'https://zenodo.org/records/8187943/files/mean_and_std_healpix.fits'},
    })

# default distance modulus grid for the table, 10 pc to 100 kpc
DM_GRID = np.round(np.arange(0.0,20.0+1E-8,0.05),5)

# User-Agent for the requests to the map archives (Harvard Dataverse
# rejects the default python-requests agent)
HTTP_HEADERS = {'User-Agent':'uberMS-dustprior'}

# line-of-sight profiles and 2D map query objects read in this session
_PROFILE_CACHE = {}
_QUERY2D_CACHE = {}

def clear_dustmap_cache():
    """Forget the line-of-sight profiles and 2D maps kept in memory."""
    _PROFILE_CACHE.clear()
    _QUERY2D_CACHE.clear()

class DustMapUnavailable(Exception):
    """A dust map could not be read locally or online."""
    pass

# ------------------------------------------------------------------------
# file access, local or remote

def _data_dir():
    from dustmaps.std_paths import data_dir
    return data_dir()

def _localfile(name):
    for ff in MAPFILES[name]['local']:
        path = os.path.join(_data_dir(),ff)
        if os.path.isfile(path):
            return path
    return None

def _remoteurl(name):
    """Direct download URL of the online map file, supports range requests."""
    import requests
    info = MAPFILES[name]
    if 'url' in info.keys():
        return info['url']
    doi,fname = info['dataverse']
    r = requests.get(
        f'https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:{doi}',
        headers=HTTP_HEADERS,timeout=60)
    r.raise_for_status()
    fileid = None
    for ff in r.json()['data']['latestVersion']['files']:
        if ff['dataFile']['filename'] == fname:
            fileid = ff['dataFile']['id']
    if fileid is None:
        raise DustMapUnavailable(f'{fname} not found in doi:{doi}')
    # Dataverse redirects to a signed S3 URL, which supports range requests
    r = requests.get(f'https://dataverse.harvard.edu/api/access/datafile/{fileid}',
                     headers=HTTP_HEADERS,allow_redirects=False,timeout=60)
    if r.status_code in (301,302,303,307,308):
        return r.headers['location']
    r.raise_for_status()
    return f'https://dataverse.harvard.edu/api/access/datafile/{fileid}'

def _resolve(name, source):
    """('local', path) or ('remote', url) for a 3D map."""
    if source not in ('auto','local','remote'):
        raise ValueError(f"[uberMS] dust map source must be 'auto', 'local', or 'remote', got {source}")
    if source in ('auto','local'):
        path = _localfile(name)
        if path is not None:
            return 'local',path
        if source == 'local':
            raise DustMapUnavailable(
                f'{name} is not in the dustmaps data directory {_data_dir()}')
    try:
        return 'remote',_remoteurl(name)
    except DustMapUnavailable:
        raise
    except Exception as e:
        raise DustMapUnavailable(f'{name} could not be reached online ({type(e).__name__}: {e})')

def _open_h5(kind, loc):
    import h5py
    if kind == 'local':
        return h5py.File(loc,'r')
    import fsspec
    fh = fsspec.filesystem('http',headers=HTTP_HEADERS).open(
        loc,mode='rb',block_size=2**16,cache_type='bytes')
    return h5py.File(fh,'r')

class _RangeReader(object):
    """Read byte ranges from a local file or, with HTTP range requests, a URL."""
    def __init__(self, kind, loc, maxranges=100, nthreads=4, mergegap=8192, maxretry=8):
        self.kind,self.loc = kind,loc
        self.maxranges,self.nthreads = maxranges,nthreads
        self.mergegap,self.maxretry = mergegap,maxretry
        if kind == 'remote':
            import requests
            self.session = requests.Session()
            self.session.headers.update(HTTP_HEADERS)

    def read(self, ranges):
        """ranges: list of (offset, nbytes), returns list of bytes."""
        if self.kind == 'local':
            out = []
            with open(self.loc,'rb') as f:
                for off,n in ranges:
                    f.seek(off); out.append(f.read(n))
            return out

        # remote: merge ranges closer than mergegap bytes (fewer, slightly
        # larger reads), batch several ranges per request, and run the
        # requests in parallel
        order = sorted(range(len(ranges)),key=lambda i: ranges[i][0])
        merged = []
        for i in order:
            off,n = ranges[i]
            if merged and off <= merged[-1][1] + self.mergegap:
                merged[-1][1] = max(merged[-1][1],off+n)
            else:
                merged.append([off,off+n])
        batches = [merged[i:i+self.maxranges] for i in range(0,len(merged),self.maxranges)]

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.nthreads) as ex:
            results = list(ex.map(self._getbatch,batches))
        blocks = {}
        for res in results:
            blocks.update(res)
        starts = np.array(sorted(blocks.keys()))
        out = []
        for off,n in ranges:
            s = starts[np.searchsorted(starts,off,side='right')-1]
            out.append(blocks[s][off-s:off-s+n])
        return out

    def _getbatch(self, batch):
        hdr = {'Range':'bytes=' + ','.join(f'{a}-{b-1}' for a,b in batch)}
        r = None
        for attempt in range(self.maxretry):
            try:
                r = self.session.get(self.loc,headers=hdr,timeout=120)
            except Exception:
                if attempt == self.maxretry-1:
                    raise
                time.sleep(2.0**attempt)
                continue
            if r.status_code == 206:
                break
            if r.status_code in (429,500,502,503,504):
                # rate limited or temporarily unavailable: wait as asked by
                # the server, or back off exponentially
                try:
                    wait = float(r.headers.get('Retry-After',''))
                except ValueError:
                    wait = 2.0**attempt
                time.sleep(min(max(wait,1.0),120.0))
                continue
            break
        if r.status_code == 429:
            raise DustMapUnavailable(
                'the map server is rate limiting requests (HTTP 429), try again later or '
                'download the map with dustmaps for large batches of stars')
        if r.status_code != 206:
            raise DustMapUnavailable(f'server did not return the requested ranges (HTTP {r.status_code})')
        body = r.content
        out = {}
        if body.lstrip()[:2] == b'--':
            # multipart/byteranges, the server may coalesce nearby ranges
            # into one part, and some servers use bare \n line endings
            for m in re.finditer(rb'Content-Range: bytes (\d+)-(\d+)/\d+\r?\n\r?\n',body,flags=re.IGNORECASE):
                a,b = int(m.group(1)),int(m.group(2))
                out[a] = body[m.end():m.end()+(b-a+1)]
        else:
            # a single part, its range is in the response header
            m = re.match(r'bytes (\d+)-(\d+)/',r.headers.get('Content-Range',''))
            if m is not None:
                out[int(m.group(1))] = body
        # every requested range must be inside a returned part
        starts = np.array(sorted(out.keys()))
        for a,b in batch:
            i = np.searchsorted(starts,a,side='right')-1
            if (i < 0) or (starts[i]+len(out[starts[i]]) < b):
                raise DustMapUnavailable('could not parse the range response from the server')
        return out

# ------------------------------------------------------------------------
# readers for the 3D maps, each returns the profile of one line of sight

def _find_sorted(arr, target, key=lambda x: x):
    """Binary search in a sorted h5py dataset without reading all of it."""
    lo,hi = 0,arr.shape[0]
    while lo < hi:
        mid = (lo+hi)//2
        if key(arr[mid]) < target:
            lo = mid+1
        else:
            hi = mid
    if (lo < arr.shape[0]) and (key(arr[lo]) == target):
        return lo
    return None

def _profile_bayestar(kind, loc, l, b):
    """
    Bayestar19: the pixel containing (l,b) at its HEALPix resolution, all
    posterior samples at the distance bin edges, and the reliable range.
    The pixel table is sorted by (nside, healpix_index).
    """
    import healpy as hp
    with _open_h5(kind,loc) as f:
        pinfo = f['pixel_info']
        dm = np.asarray(pinfo.attrs['DM_bin_edges'],dtype=float)
        idx = None
        for nside in [64,128,256,512,1024,2048]:
            ipix = int(hp.ang2pix(nside,l,b,lonlat=True,nest=True))
            idx = _find_sorted(pinfo,(nside,ipix),key=lambda r: (int(r['nside']),int(r['healpix_index'])))
            if idx is not None:
                break
        if idx is None:
            return None
        row = pinfo[idx]
        samples = np.asarray(f['samples'][idx,:,:],dtype=float)
    dmmin,dmmax = float(row['DM_reliable_min']),float(row['DM_reliable_max'])
    # as in dustmaps, NaN reliable limits mean no reliable distances
    valid = np.isfinite(dmmin) & np.isfinite(dmmax) & (dm >= dmmin) & (dm <= dmmax)
    conv = AV_CONVERSION['bayestar']
    return ({
        'dm':dm,'Av_mean':conv*samples.mean(axis=0),'Av_std':conv*samples.std(axis=0),
        'valid':valid & np.all(np.isfinite(samples),axis=0),'converged':bool(row['converged']),
        })

def _profile_decaps(kind, loc, l, b):
    """
    DECaPS: the NSIDE=8192 pixel containing (l,b), the mean (or the
    samples, if the local file has them) at the distance bin edges, and the
    reliable range.
    """
    import healpy as hp
    with _open_h5(kind,loc) as f:
        pinfo = f['pixel_info']
        dm = np.asarray(pinfo.attrs['DM_bin_edges'],dtype=float)
        nside = int(pinfo.attrs['nside'])
        ipix = int(hp.ang2pix(nside,l,b,lonlat=True,nest=True))
        idx = _find_sorted(pinfo['healpix_index'],ipix,key=lambda x: int(x))
        if idx is None:
            return None
        dmmin = float(pinfo['DM_reliable_min'][idx])
        dmmax = float(pinfo['DM_reliable_max'][idx])
        converged = bool(pinfo['converged'][idx]) if 'converged' in pinfo.keys() else True
        if 'samples' in f.keys():
            samples = np.asarray(f['samples'][idx,...],dtype=float).reshape(-1,dm.size)
            mean,std = samples.mean(axis=0),samples.std(axis=0)
        else:
            mean = np.asarray(f['mean'][idx,...],dtype=float).reshape(dm.size)
            std = None
    valid = np.isfinite(dmmin) & np.isfinite(dmmax) & (dm >= dmmin) & (dm <= dmmax) & np.isfinite(mean)
    conv = AV_CONVERSION['decaps']
    return ({
        'dm':dm,'Av_mean':conv*mean,'Av_std':None if std is None else conv*std,
        'valid':valid,'converged':converged,
        })

_FITS_LAYOUT = {}

def _fits_layout(reader):
    """Offsets and shapes of the extensions of a FITS file, from the headers."""
    if reader.loc in _FITS_LAYOUT.keys():
        return _FITS_LAYOUT[reader.loc]
    exts = {}
    off = 0
    for _ in range(20):
        cards,hlen = [],0
        while True:
            try:
                raw = reader.read([(off+hlen,2880)])[0]
            except Exception:
                raw = b''
            if len(raw) < 2880:
                break
            hlen += 2880
            raw = raw.decode('ascii','replace')
            cards += [raw[i:i+80] for i in range(0,2880,80)]
            if any(c.startswith('END') for c in cards[-36:]):
                break
        if not cards:
            break
        kv = {}
        for c in cards:
            if c[8:10] == '= ':
                kv[c[:8].strip()] = c[10:].split('/')[0].strip().strip("'").strip()
        nax = int(kv.get('NAXIS',0))
        size = abs(int(kv['BITPIX']))//8 if nax > 0 else 0
        shape = []
        for i in range(1,nax+1):
            size *= int(kv[f'NAXIS{i}']); shape.append(int(kv[f'NAXIS{i}']))
        size += int(kv.get('PCOUNT',0) or 0)
        exts[kv.get('EXTNAME','PRIMARY')] = ({
            'data':off+hlen,'shape':shape,'bitpix':int(kv['BITPIX']),'header':kv})
        off += hlen + ((size+2879)//2880)*2880
    _FITS_LAYOUT[reader.loc] = exts
    return exts

def _profile_edenhofer(kind, loc, l, b):
    """
    Edenhofer+2023: integrate the extinction density along the line of
    sight, starting from the integrated extinction of the inner 68.8 pc.
    The density is interpolated between the 4 nearest HEALPix pixels in log
    space, as in dustmaps. The std of the integrated extinction adds the
    voxel stds in quadrature (independent voxels); against the posterior
    samples of the map this overestimates the std by ~2x, while assuming
    fully correlated voxels overestimates it by ~15x. Only the bytes of
    these pixels are read.
    """
    import healpy as hp
    reader = _RangeReader(kind,loc)
    exts = _fits_layout(reader)
    try:
        emean,estd = exts['MEAN'],exts['STD.']
        eedge = exts['RADIAL PIXEL BOUNDARIES']
        einner = [kk for kk in exts.keys() if kk.startswith('MEAN OF INTEGRATED INNER')][0]
        einner_std = [kk for kk in exts.keys() if kk.startswith('STD. OF INTEGRATED INNER')][0]
    except (KeyError,IndexError):
        raise DustMapUnavailable(f'unexpected file layout for the Edenhofer map: {list(exts.keys())}')
    npix,nrad = emean['shape']
    nside = int(emean['header'].get('NSIDE',hp.npix2nside(npix)))
    nest = emean['header'].get('ORDERING','NEST').upper() == 'NEST'

    pix,wgt = hp.get_interp_weights(nside,l,b,nest=nest,lonlat=True)
    pix,wgt = np.asarray(pix,dtype=int),np.asarray(wgt,dtype=float)

    ranges = [(emean['data']+(k*npix+p)*4,4) for k in range(nrad) for p in pix]
    ranges += [(estd['data']+(k*npix+p)*4,4) for k in range(nrad) for p in pix]
    ranges += [(exts[einner]['data']+p*4,4) for p in pix]
    ranges += [(exts[einner_std]['data']+p*4,4) for p in pix]
    ranges += [(eedge['data'],4*(nrad+1))]
    raw = reader.read(ranges)

    vals = np.frombuffer(b''.join(raw[:-1]),dtype='>f4').astype(float)
    n4 = nrad*pix.size
    dens = vals[:n4].reshape(nrad,pix.size)
    dstd = vals[n4:2*n4].reshape(nrad,pix.size)
    inner = vals[2*n4:2*n4+pix.size]
    inner_std = vals[2*n4+pix.size:2*n4+2*pix.size]
    edges = np.frombuffer(raw[-1],dtype='>f4').astype(float)

    # interpolate as dustmaps: density in log space, std in variance space
    dens = np.exp(np.sum(wgt*np.log(dens),axis=1))
    dstd = np.sqrt(np.sum(wgt*dstd**2,axis=1))
    inner = np.sum(wgt*inner)
    inner_std = np.sqrt(np.sum(wgt*inner_std**2))

    dr = np.diff(edges)
    mean = np.concatenate([[inner],inner+np.cumsum(dens*dr)])
    std = np.sqrt(np.concatenate([[inner_std**2],inner_std**2+np.cumsum((dstd*dr)**2)]))
    conv = AV_CONVERSION['edenhofer2023']
    return ({
        'dm':5.0*np.log10(edges)-5.0,'Av_mean':conv*mean,'Av_std':conv*std,
        'valid':np.isfinite(mean),'converged':True,
        })

_READERS = {'bayestar':_profile_bayestar,'decaps':_profile_decaps,'edenhofer2023':_profile_edenhofer}

def _incoverage(name, l, b):
    """
    Footprint check so that maps that cannot cover (l,b) are not read. The
    map itself is the final word on coverage.
    """
    if name == 'bayestar':
        # Pan-STARRS 1 footprint, dec > -30 deg
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        dec = SkyCoord(l=l*u.deg,b=b*u.deg,frame='galactic').icrs.dec.deg
        return dec > -30.5
    if name == 'decaps':
        # southern Galactic plane, 239 < l < 6, |b| < 10
        ll = l % 360.0
        return ((ll > 238.5) or (ll < 6.5)) and (abs(b) < 10.5)
    return True

def query_profile(name, l, b, source='auto', verbose=False):
    """
    A_V profile of a 3D map along (l,b). Returns None if the map has no
    pixel at (l,b), otherwise a dict with the distance modulus grid of the
    map ('dm'), the mean and std of A_V ('Av_mean', 'Av_std', the latter
    None if the map gives no uncertainty), the points inside the reliable
    range of the map ('valid'), and where it was read from ('source').
    """
    if name not in MAPS3D:
        raise ValueError(f'[uberMS] {name} is not a 3D dust map, 3D maps are {MAPS3D}')
    key = (name,source,float(l),float(b))
    if key in _PROFILE_CACHE.keys():
        return _PROFILE_CACHE[key]
    kind,loc = _resolve(name,source)
    t0 = time.time()
    try:
        prof = _READERS[name](kind,loc,float(l),float(b))
    except DustMapUnavailable:
        raise
    except Exception as e:
        raise DustMapUnavailable(f'{name} could not be read from the {kind} file ({type(e).__name__}: {e})')
    if prof is not None:
        prof['map'] = name
        prof['source'] = kind
        prof['readtime'] = time.time()-t0
        if verbose:
            print(f'[uberMS]     {name}: read line of sight from {kind} file in {prof["readtime"]:.1f} s')
    _PROFILE_CACHE[key] = prof
    return prof

def query_2d(name, l, b):
    """Total A_V along (l,b) from a 2D map in the dustmaps data directory."""
    if name not in _QUERY2D_CACHE.keys():
        if name == 'sfd':
            from dustmaps.sfd import SFDQuery
            _QUERY2D_CACHE[name] = SFDQuery()
        elif name == 'csfd':
            from dustmaps.csfd import CSFDQuery
            _QUERY2D_CACHE[name] = CSFDQuery()
        else:
            raise ValueError(f'[uberMS] {name} is not a 2D dust map, 2D maps are {MAPS2D}')
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    val = float(np.asarray(_QUERY2D_CACHE[name](SkyCoord(l=l*u.deg,b=b*u.deg,frame='galactic'))))
    return AV_CONVERSION[name]*val

# ------------------------------------------------------------------------
# combine the maps and build the prior table

def _ongrid(prof, dm_grid, taper):
    """
    Interpolate a map profile onto dm_grid over its reliable range. The
    weight of the map tapers linearly to ~0 within taper mag of the edges
    of its reliable range, so that the combined profile changes smoothly
    where maps start or stop overlapping.
    """
    ng = dm_grid.size
    m,s,w = np.full(ng,np.nan),np.zeros(ng),np.zeros(ng)
    valid = prof['valid']
    if valid.sum() < 2:
        return m,s,w
    dm = prof['dm'][valid]
    lo,hi = dm.min(),dm.max()
    inside = (dm_grid >= lo) & (dm_grid <= hi)
    m[inside] = np.interp(dm_grid[inside],dm,prof['Av_mean'][valid])
    if prof['Av_std'] is not None:
        s[inside] = np.interp(dm_grid[inside],dm,prof['Av_std'][valid])
    edge = np.minimum(dm_grid-lo,hi-dm_grid)
    if taper > 0:
        w[inside] = np.clip(edge[inside]/taper,1E-3,1.0)
    else:
        w[inside] = 1.0
    return m,s,w

def combine_profiles(profiles, dm_grid=DM_GRID, taper=0.25):
    """
    Combine the A_V profiles of several 3D maps on dm_grid.

    Where maps overlap, the mean is the weighted mean of the maps, and the
    variance is the weighted mean of the variances reported by each map
    plus the (unbiased, weighted) variance between the maps. Where only one
    map exists, its own uncertainty is used.

    Returns mean, variance, and a boolean array of where any map is valid.
    """
    if len(profiles) == 0:
        ng = dm_grid.size
        return np.full(ng,np.nan),np.zeros(ng),np.zeros(ng,dtype=bool)

    M,S,W = [],[],[]
    for prof in profiles:
        m,s,w = _ongrid(prof,dm_grid,taper)
        M.append(m); S.append(s); W.append(w)
    M,S,W = np.array(M),np.array(S),np.array(W)
    M0 = np.where(W > 0,M,0.0)

    V1 = W.sum(axis=0)
    V2 = (W**2).sum(axis=0)
    has = V1 > 0
    V1s = np.where(has,V1,1.0)

    mean = (W*M0).sum(axis=0)/V1s
    var_within = (W*S**2).sum(axis=0)/V1s
    # unbiased weighted variance between the maps, 0 for a single map
    denom = V1s-V2/V1s
    nmap = (W > 0).sum(axis=0)
    var_between = np.where(
        (nmap > 1) & (denom > 1E-8),
        (W*(M0-mean)**2).sum(axis=0)/np.where(denom > 1E-8,denom,1.0),
        0.0)

    mean = np.where(has,mean,np.nan)
    return mean,var_within+var_between,has

def build_av3d_table(
    l, b, maps='auto', source='auto', map2d='sfd', profiles=None, Av2d=None,
    near='flat', sigma=0.05, fracsigma=0.1, fracsigma_extrap=0.2,
    ramp_dm=1.0, taper=0.25, dust_scaleheight=125.0, max_pathlength=5000.0,
    Av_ll=1E-6, Av_ul=None, dm_grid=None, verbose=True):
    """
    Tabulate the A_V prior versus distance along (l,b).

    The prior at distance d is flat between Av_lo(d) and Av_hi(d), with
    Gaussian tails of width Av_sigma(d) (see advancedpriors.PlateauNormal).
    Where the 3D maps are reliable, Av_lo = Av_hi = the map mean, i.e., a
    truncated normal.

    Parameters
    ----------
    l, b : float
        Galactic longitude and latitude in degrees.
    maps : list of str or 'auto'
        3D maps to combine (from MAPS3D). 'auto' uses every 3D map whose
        footprint covers (l,b) and that can be read; maps that cannot be
        read are skipped. A map named explicitly must be readable.
    source : 'auto', 'local', or 'remote'
        Where to read the 3D maps. 'auto' uses the local file in the
        dustmaps data directory if it exists, and otherwise reads just this
        line of sight from the online archive with HTTP range requests
        (Zenodo for Edenhofer+2023, Harvard Dataverse for Bayestar19 and the
        DECaPS mean map), without downloading the map.
    map2d : str or None
        2D map (from MAPS2D, read from the dustmaps data directory) giving
        the total A_V along the line of sight, used beyond the reliable
        range of the 3D maps. None holds the A_V of the 3D maps constant
        beyond their range.
    profiles : list of dict, optional
        Precomputed 3D map profiles (see query_profile), skips the queries.
    Av2d : float, optional
        Total A_V along the line of sight, skips the 2D map query.
    near : 'flat' or 'linear'
        Prior closer than the first reliable distance d0 of the 3D maps.
        'flat' (default): flat from 0 to Av(d0), with a Gaussian fall-off
        of width sigma(d0) above Av(d0). 'linear': Av rises linearly in
        distance from 0, with extra uncertainty.
    sigma : float
        Absolute uncertainty floor on A_V in mag.
    fracsigma : float
        Fractional uncertainty on the A_V of the maps.
    fracsigma_extrap : float
        Extra fractional uncertainty where A_V is extrapolated: beyond the
        3D maps (on the 2D total), between the Sun and the 3D maps for
        near='linear', and for lines of sight without 3D coverage.
    ramp_dm : float
        Beyond the 3D maps A_V ramps linearly in distance modulus from the
        last 3D value to the 2D total over ramp_dm mag.
    taper : float
        Width in mag of the weight taper at the edges of each 3D map (see
        combine_profiles), and of the transition from the flat prior to the
        map at d0.
    dust_scaleheight, max_pathlength : float
        Without 3D coverage, A_V(d) = A_V,2D (1 - exp(-d/L)) for a dust slab
        with scale height dust_scaleheight in pc, L = h/|sin b| capped at
        max_pathlength pc.
    Av_ll, Av_ul : float
        Support of the A_V prior. Av_ul defaults to the larger of 5 and the
        maximum of Av_hi + 5 sigma.
    dm_grid : array, optional
        Distance modulus grid of the table, defaults to DM_GRID.

    Returns
    -------
    dict with 'dist_grid' (pc, starting at 0), 'Av_lo', 'Av_hi', 'Av_sigma',
    'Av_mean' (center of the prior), the support 'Av_ll','Av_ul', and
    metadata on the maps used.
    """
    if near not in ('flat','linear'):
        raise ValueError(f"[uberMS] near must be 'flat' or 'linear', got {near}")
    dm_grid = DM_GRID if dm_grid is None else np.asarray(dm_grid,dtype=float)
    dgrid = 10.0**(dm_grid/5.0+1.0)

    # read the 3D maps
    skipped = {}
    if profiles is None:
        if maps == 'auto':
            mapnames,explicit = list(MAPS3D),False
        else:
            mapnames,explicit = ([maps] if isinstance(maps,str) else list(maps)),True
        profiles = []
        for name in mapnames:
            if name not in MAPS3D:
                raise ValueError(f'[uberMS] {name} is not a 3D dust map, 3D maps are {MAPS3D}')
            if not _incoverage(name,l,b):
                skipped[name] = 'outside footprint'
                continue
            try:
                prof = query_profile(name,l,b,source=source,verbose=verbose)
            except DustMapUnavailable as e:
                if explicit:
                    raise
                skipped[name] = f'not available, {e}'
                continue
            if (prof is None) or (prof['valid'].sum() < 2):
                skipped[name] = 'no reliable distances along this line of sight'
                continue
            if (not prof['converged']) and verbose:
                print(f'[uberMS] WARNING: {name} line-of-sight fit did not converge at l={l}, b={b}')
            profiles.append(prof)
    else:
        profiles = [pp for pp in profiles if pp['valid'].sum() >= 2]

    # total extinction from the 2D map
    if (Av2d is None) and (map2d is not None):
        try:
            Av2d = query_2d(map2d,l,b)
        except (OSError,FileNotFoundError) as e:
            print(f'[uberMS] WARNING: 2D map {map2d} is not installed ({type(e).__name__}), beyond the '
                  f'3D maps Av is held at its last value; install it with dustmaps.{map2d}.fetch()')
            map2d = None

    mean,mapvar,has = combine_profiles(profiles,dm_grid=dm_grid,taper=taper)
    extra = np.zeros(dm_grid.size)
    idx = np.arange(dm_grid.size)
    nearmask = np.zeros(dm_grid.size,dtype=bool)

    if has.any():
        i0,i1 = idx[has][0],idx[has][-1]

        # gaps between the maps, interpolate in distance modulus
        gap = (~has) & (idx > i0) & (idx < i1)
        if gap.any():
            mean[gap] = np.interp(dm_grid[gap],dm_grid[has],mean[has])
            mapvar[gap] = np.interp(dm_grid[gap],dm_grid[has],mapvar[has])

        # closer than the 3D maps
        nearmask = idx < i0
        x = dgrid[nearmask]/dgrid[i0]
        mapvar[nearmask] = mapvar[i0]
        if near == 'linear':
            # dust uniformly distributed between the Sun and d0, the extra
            # uncertainty is largest halfway and continuous at both ends
            mean[nearmask] = mean[i0]*x
            extra[nearmask] = fracsigma_extrap*mean[i0]*2.0*np.sqrt(x*(1.0-x))
        else:
            mean[nearmask] = mean[i0]

        # beyond the 3D maps: ramp to the 2D total
        far = idx > i1
        Avfar = mean[i1] if Av2d is None else max(Av2d,mean[i1])
        t = np.clip((dm_grid[far]-dm_grid[i1])/ramp_dm,0.0,1.0) if ramp_dm > 0 else np.ones(far.sum())
        mean[far] = mean[i1]+t*(Avfar-mean[i1])
        mapvar[far] = mapvar[i1]
        extra[far] = fracsigma_extrap*mean[far]*t

        dmrange = [float(dm_grid[i0]),float(dm_grid[i1])]
    else:
        if Av2d is None:
            raise ValueError(
                f'[uberMS] no 3D dust map covers l={l}, b={b} and no 2D map is available; '
                f'skipped maps: {skipped}')
        # dust slab model scaled to the 2D total
        sinb = abs(np.sin(np.deg2rad(b)))
        L = max_pathlength if sinb*max_pathlength < dust_scaleheight else dust_scaleheight/sinb
        mean = Av2d*(1.0-np.exp(-dgrid/L))
        mapvar = np.zeros(dm_grid.size)
        extra = fracsigma_extrap*mean
        dmrange = None

    # cumulative extinction cannot decrease with distance
    mean = np.maximum.accumulate(mean)
    sig = np.sqrt(mapvar + sigma**2 + (fracsigma*mean)**2 + extra**2)

    # plateau edges: equal to the mean except for the flat prior near the
    # Sun, whose lower edge rises from 0 to Av(d0) over the last taper mag
    # before d0 so that the prior is continuous in distance
    Av_hi = mean.copy()
    Av_lo = mean.copy()
    if (near == 'flat') and nearmask.any():
        dm0 = dm_grid[idx[has][0]]
        ramp = np.clip((dm_grid[nearmask]-(dm0-taper))/taper,0.0,1.0) if taper > 0 else np.zeros(nearmask.sum())
        Av_lo[nearmask] = mean[nearmask]*ramp

    # start the table at the Sun
    dist_grid = np.concatenate([[0.0],dgrid])
    Av_lo = np.concatenate([[0.0 if near == 'linear' else Av_lo[0]],Av_lo])
    Av_hi = np.concatenate([[0.0 if near == 'linear' else Av_hi[0]],Av_hi])
    sig = np.concatenate([[sig[0]],sig])

    if Av_ul is None:
        Av_ul = float(max(5.0,np.ceil(np.max(Av_hi+5.0*sig))))
    if Av_ll >= Av_ul:
        raise ValueError(f'[uberMS] 3D Av prior support [{Av_ll}, {Av_ul}] is empty')

    table = ({
        'dist_grid':dist_grid,
        'Av_lo':Av_lo,
        'Av_hi':Av_hi,
        'Av_mean':0.5*(Av_lo+Av_hi),
        'Av_sigma':sig,
        'Av_ll':float(Av_ll),
        'Av_ul':float(Av_ul),
        'l':float(l),'b':float(b),
        'near':near,
        'maps':[pp['map'] for pp in profiles],
        'map_source':{pp['map']:pp.get('source','given') for pp in profiles},
        'map_dmrange':{pp['map']:[float(pp['dm'][pp['valid']].min()),float(pp['dm'][pp['valid']].max())]
                       for pp in profiles},
        'dmrange3d':dmrange,
        'map2d':map2d,
        'Av2d':None if Av2d is None else float(Av2d),
        'skipped':skipped,
        })

    if verbose:
        _report(table)
    return table

def _report(tab):
    print(f"[uberMS] ... 3D Av prior along l={tab['l']:.4f}, b={tab['b']:.4f}")
    if len(tab['maps']) > 0:
        for kk,(lo,hi) in tab['map_dmrange'].items():
            print(f"[uberMS]     {kk} ({tab['map_source'][kk]}): reliable "
                  f"{10.0**(lo/5.0+1.0):.0f}-{10.0**(hi/5.0+1.0):.0f} pc")
    else:
        print('[uberMS]     no 3D map coverage, using a dust slab scaled to the 2D map')
    for kk,vv in tab['skipped'].items():
        print(f'[uberMS]     {kk}: skipped, {vv}')
    if tab['Av2d'] is not None:
        print(f"[uberMS]     {tab['map2d']} total Av = {tab['Av2d']:.3f}")
    if tab['dmrange3d'] is not None and tab['near'] == 'flat':
        d0 = 10.0**(tab['dmrange3d'][0]/5.0+1.0)
        A0 = np.interp(d0,tab['dist_grid'],tab['Av_hi'])
        print(f'[uberMS]     closer than {d0:.0f} pc: flat in Av from 0 to {A0:.3f}')
    dd = np.array([100.0,500.0,1000.0,3000.0,10000.0])
    lo = np.interp(dd,tab['dist_grid'],tab['Av_lo'])
    hi = np.interp(dd,tab['dist_grid'],tab['Av_hi'])
    sg = np.interp(dd,tab['dist_grid'],tab['Av_sigma'])
    def fmt(lo_i,hi_i,s_i):
        if hi_i-lo_i > 1E-6:
            return f'[{lo_i:.3f},{hi_i:.3f}]+-{s_i:.3f}'
        return f'{hi_i:.3f}+-{s_i:.3f}'
    print('[uberMS]     ' + ', '.join(
        f'Av({d_i:.0f} pc) = {fmt(l_i,h_i,s_i)}' for d_i,l_i,h_i,s_i in zip(dd,lo,hi,sg)))
    print(f"[uberMS]     Av support [{tab['Av_ll']:.3g}, {tab['Av_ul']:.3g}]")

def dustprior_setup(priors, verbose=True):
    """
    Build the table for a 3D dust map prior on Av, returns a copy of priors
    with priors['Av'] = ['3D', table]. Priors without a 3D Av prior, or with
    a table already built, are returned unchanged.
    """
    if ('Av' not in priors.keys()) or (priors['Av'][0] != '3D'):
        return priors
    info = priors['Av'][1]
    if 'dist_grid' in info.keys():
        return priors
    if ('l' not in info.keys()) or ('b' not in info.keys()):
        raise ValueError("[uberMS] 3D Av prior requires 'l' and 'b', e.g., priors['Av'] = ['3D',{'l':l,'b':b}]")

    kw = dict(info)
    l,b = kw.pop('l'),kw.pop('b')
    kw.setdefault('verbose',verbose)
    priors = dict(priors)
    priors['Av'] = ['3D',build_av3d_table(l,b,**kw)]

    if verbose and ('Rv' in priors.keys()) and (priors['Rv'][0] != 'fixed'):
        print('[uberMS] NOTE: the 3D Av prior converts the dust maps to A_V assuming Rv ~ 3.1')
    return priors
