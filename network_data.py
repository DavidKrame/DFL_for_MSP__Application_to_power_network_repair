"""
Network Data Module
===================
Provides the toy (5-bus) network and a loader for PGLib-OPF test cases.

Toy network topology:
    Node 1 (Gen1) -- Line 1 -- Node 2 (Load1)
         |                          |
      Line 2                     Line 4
         |                          |
    Node 3 (Load2) -- Line 3 -- Node 4 (Gen2)
                                   |
                                Line 5
                                   |
                              Node 5 (Load3)

PGLib-OPF test cases are downloaded on first use and cached in
~/.cache/pglib_opf/. Loading them requires pandapower and matpowercaseframes,
which pandapower uses to parse .m files (both are in requirements.txt).
"""

import re
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# PGLib-OPF case registry
# ---------------------------------------------------------------------------

PGLIB_BASE_URL = (
    "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/"
)

# short_key -> (n_buses, filename_stem)
# Keys and stems match the case files on the pglib-opf master branch
# (https://github.com/power-grid-lib/pglib-opf). PGLib has no case9_ieee; the
# smallest IEEE case is case14_ieee.
_PGLIB_CASES = {
    'case3_lmbd':         (3,     'pglib_opf_case3_lmbd'),
    'case5_pjm':          (5,     'pglib_opf_case5_pjm'),
    'case14_ieee':        (14,    'pglib_opf_case14_ieee'),
    'case24_ieee_rts':    (24,    'pglib_opf_case24_ieee_rts'),
    'case30_as':          (30,    'pglib_opf_case30_as'),
    'case30_ieee':        (30,    'pglib_opf_case30_ieee'),
    'case39_epri':        (39,    'pglib_opf_case39_epri'),
    'case57_ieee':        (57,    'pglib_opf_case57_ieee'),
    'case60_c':           (60,    'pglib_opf_case60_c'),
    'case73_ieee_rts':    (73,    'pglib_opf_case73_ieee_rts'),
    'case89_pegase':      (89,    'pglib_opf_case89_pegase'),
    'case118_ieee':       (118,   'pglib_opf_case118_ieee'),
    'case162_ieee_dtc':   (162,   'pglib_opf_case162_ieee_dtc'),
    'case179_goc':        (179,   'pglib_opf_case179_goc'),
    'case197_snem':       (197,   'pglib_opf_case197_snem'),
    'case200_activ':      (200,   'pglib_opf_case200_activ'),
    'case240_pserc':      (240,   'pglib_opf_case240_pserc'),
    'case300_ieee':       (300,   'pglib_opf_case300_ieee'),
    'case500_goc':        (500,   'pglib_opf_case500_goc'),
    'case588_sdet':       (588,   'pglib_opf_case588_sdet'),
    'case793_goc':        (793,   'pglib_opf_case793_goc'),
    'case1354_pegase':    (1354,  'pglib_opf_case1354_pegase'),
    'case1803_snem':      (1803,  'pglib_opf_case1803_snem'),
    'case1888_rte':       (1888,  'pglib_opf_case1888_rte'),
    'case1951_rte':       (1951,  'pglib_opf_case1951_rte'),
    'case2000_goc':       (2000,  'pglib_opf_case2000_goc'),
    'case2312_goc':       (2312,  'pglib_opf_case2312_goc'),
    'case2383wp_k':       (2383,  'pglib_opf_case2383wp_k'),
    'case2742_goc':       (2742,  'pglib_opf_case2742_goc'),
    'case2736sp_k':       (2736,  'pglib_opf_case2736sp_k'),
    'case2737sop_k':      (2737,  'pglib_opf_case2737sop_k'),
    'case2746wop_k':      (2746,  'pglib_opf_case2746wop_k'),
    'case2746wp_k':       (2746,  'pglib_opf_case2746wp_k'),
    'case2848_rte':       (2848,  'pglib_opf_case2848_rte'),
    'case2853_sdet':      (2853,  'pglib_opf_case2853_sdet'),
    'case2868_rte':       (2868,  'pglib_opf_case2868_rte'),
    'case2869_pegase':    (2869,  'pglib_opf_case2869_pegase'),
    'case3012wp_k':       (3012,  'pglib_opf_case3012wp_k'),
    'case3022_goc':       (3022,  'pglib_opf_case3022_goc'),
    'case3120sp_k':       (3120,  'pglib_opf_case3120sp_k'),
    'case3375wp_k':       (3375,  'pglib_opf_case3375wp_k'),
    'case3970_goc':       (3970,  'pglib_opf_case3970_goc'),
    'case4020_goc':       (4020,  'pglib_opf_case4020_goc'),
    'case4601_goc':       (4601,  'pglib_opf_case4601_goc'),
    'case4619_goc':       (4619,  'pglib_opf_case4619_goc'),
    'case4661_sdet':      (4661,  'pglib_opf_case4661_sdet'),
    'case4837_goc':       (4837,  'pglib_opf_case4837_goc'),
    'case4917_goc':       (4917,  'pglib_opf_case4917_goc'),
    'case5658_epigrids':  (5658,  'pglib_opf_case5658_epigrids'),
    'case6468_rte':       (6468,  'pglib_opf_case6468_rte'),
    'case6470_rte':       (6470,  'pglib_opf_case6470_rte'),
    'case6495_rte':       (6495,  'pglib_opf_case6495_rte'),
    'case6515_rte':       (6515,  'pglib_opf_case6515_rte'),
    'case7336_epigrids':  (7336,  'pglib_opf_case7336_epigrids'),
    'case8387_pegase':    (8387,  'pglib_opf_case8387_pegase'),
    'case9241_pegase':    (9241,  'pglib_opf_case9241_pegase'),
    'case9591_goc':       (9591,  'pglib_opf_case9591_goc'),
    'case10000_goc':      (10000, 'pglib_opf_case10000_goc'),
    'case10192_epigrids': (10192, 'pglib_opf_case10192_epigrids'),
    'case10480_goc':      (10480, 'pglib_opf_case10480_goc'),
    'case13659_pegase':   (13659, 'pglib_opf_case13659_pegase'),
    'case19402_goc':      (19402, 'pglib_opf_case19402_goc'),
    'case20758_epigrids': (20758, 'pglib_opf_case20758_epigrids'),
    'case24464_goc':      (24464, 'pglib_opf_case24464_goc'),
    'case30000_goc':      (30000, 'pglib_opf_case30000_goc'),
    'case78484_epigrids': (78484, 'pglib_opf_case78484_epigrids'),
}


def _normalize_name(name):
    """Lowercase, trim, and strip a leading 'pglib_opf_' or 'pglib_' prefix."""
    name = name.lower().strip()
    for prefix in ('pglib_opf_', 'pglib_'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def _resolve_pglib_case(name):
    """Map a user-supplied name to a PGLib case key.

    Returns
    -------
    (case_key, exact) where exact=False means no case has the requested bus
    count and the case with the closest bus count was used.

    Raises ValueError if the name is not a registry key and contains no
    'case<N>'.
    """
    norm = _normalize_name(name)

    # 1. Exact key match
    if norm in _PGLIB_CASES:
        return norm, True

    # 2. Match by bus count N, parsed from the first "case<N>" in the name
    m = re.search(r'case(\d+)', norm)
    if not m:
        raise ValueError(
            f"Cannot interpret '{name}' as a PGLib network name. "
            f"Expected something like 'case14', 'case57_ieee', etc."
        )
    target_n = int(m.group(1))

    # Prefer an exact bus-count match; among ties prefer a key containing 'ieee'
    exact_n = [(k, v[0]) for k, v in _PGLIB_CASES.items() if v[0] == target_n]
    if exact_n:
        for key, _ in exact_n:
            if 'ieee' in key:
                return key, True
        return exact_n[0][0], True

    # 3. Closest by |n_buses - target_n|
    closest = min(_PGLIB_CASES.keys(),
                  key=lambda k: abs(_PGLIB_CASES[k][0] - target_n))
    return closest, False


def _download_pglib_case(case_key, cache_dir=None):
    """Return the local Path of a PGLib .m file, downloading it if not cached.

    cache_dir defaults to ~/.cache/pglib_opf.
    """
    if cache_dir is None:
        cache_dir = Path.home() / '.cache' / 'pglib_opf'
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stem = _PGLIB_CASES[case_key][1]
    filename = stem + '.m'
    local_path = cache_dir / filename

    if not local_path.exists():
        url = PGLIB_BASE_URL + filename
        print(f"  Downloading PGLib case: {filename}")
        print(f"  From: {url}")
        try:
            urllib.request.urlretrieve(url, local_path)
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to download {url}: {e}\n"
                "Check the internet connection, or manually place the .m file "
                f"at {local_path}."
            ) from e
        print(f"  Cached at: {local_path}")

    return local_path


class NetworkData:
    """Topology and parameters of a DC power network.

    The constructor builds the toy 5-bus network; from_ppc builds a network
    from a MATPOWER/PyPower case.
    """

    def __init__(self):
        self.n_buses = 5
        self.n_lines = 5
        self.n_generators = 2
        self.n_loads = 3
        self.sbase_mva = 1.0  # toy data are given directly in MW; no per-unit conversion

        # generator/load index -> bus index (0-indexed)
        self.gen_buses = [0, 3]   # buses 1 and 4
        self.load_buses = [1, 2, 4]  # buses 2, 3, 5
        self.slack_bus = 0  # bus 1 (0-indexed)

        self.gen_pmin = np.array([0.0, 0.0])          # MW
        self.gen_pmax = np.array([100.0, 150.0])       # MW
        self.gen_cost = np.array([20.0, 25.0])         # $/MWh

        self.load_nominal = np.array([60.0, 50.0, 70.0])  # MW

        # (from_bus, to_bus), 0-indexed
        self.line_connections = [
            (0, 1),  # Line 1
            (0, 2),  # Line 2
            (2, 3),  # Line 3
            (1, 3),  # Line 4
            (3, 4),  # Line 5
        ]

        self.line_susceptances = np.array([10.0, 8.0, 12.0, 9.0, 11.0])  # MW/rad (p.u. on 1 MVA)
        self.line_flow_min = np.array([-80.0, -70.0, -90.0, -75.0, -85.0])  # MW
        self.line_flow_max = np.array([80.0, 70.0, 90.0, 75.0, 85.0])   # MW

        self._build_incidence_matrix()
        self._build_susceptance_matrix()
    
    def _build_incidence_matrix(self):
        """C[i, l] = +1 if line l leaves bus i, -1 if it enters bus i, else 0."""
        self.C = np.zeros((self.n_buses, self.n_lines))
        
        for line_idx, (from_bus, to_bus) in enumerate(self.line_connections):
            self.C[from_bus, line_idx] = 1.0
            self.C[to_bus, line_idx] = -1.0
    
    def _build_susceptance_matrix(self):
        """Build the diagonal susceptance matrix B = diag(b_1, ..., b_L)."""
        self.B = np.diag(self.line_susceptances)
    
    def get_summary(self):
        """Return the network dimensions, total capacity, total load and 1-indexed slack bus."""
        return {
            'n_buses': self.n_buses,
            'n_lines': self.n_lines,
            'n_generators': self.n_generators,
            'n_loads': self.n_loads,
            'total_gen_capacity': self.gen_pmax.sum(),
            'total_load_demand': self.load_nominal.sum(),
            'slack_bus': self.slack_bus + 1,  # 1-indexed
        }
    
    def print_network_info(self):
        """Print formatted network information."""
        print("=" * 60)
        print("NETWORK INFORMATION")
        print("=" * 60)
        print(f"Number of buses:      {self.n_buses}")
        print(f"Number of lines:      {self.n_lines}")
        print(f"Number of generators: {self.n_generators}")
        print(f"Number of loads:      {self.n_loads}")
        print(f"Slack bus:            Bus {self.slack_bus + 1}")
        print()
        
        print("Generator Data:")
        print("-" * 60)
        for i in range(self.n_generators):
            bus = self.gen_buses[i] + 1  # 1-indexed
            print(f"  Gen {i+1} (Bus {bus}): "
                  f"Pmin={self.gen_pmin[i]:.1f} MW, "
                  f"Pmax={self.gen_pmax[i]:.1f} MW, "
                  f"Cost={self.gen_cost[i]:.1f} $/MWh")
        print()
        
        print("Load Data:")
        print("-" * 60)
        for i in range(self.n_loads):
            bus = self.load_buses[i] + 1  # 1-indexed
            print(f"  Load {i+1} (Bus {bus}): "
                  f"Demand={self.load_nominal[i]:.1f} MW")
        print()
        
        print("Line Data:")
        print("-" * 60)
        for i in range(self.n_lines):
            from_bus, to_bus = self.line_connections[i]
            print(f"  Line {i+1}: Bus {from_bus+1} -> Bus {to_bus+1}, "
                  f"B={self.line_susceptances[i]:.1f} MW/rad, "
                  f"Limits=[{self.line_flow_min[i]:.1f}, {self.line_flow_max[i]:.1f}] MW")
        print()
        
        print("Incidence Matrix C:")
        print("-" * 60)
        print(self.C)
        print("=" * 60)


    @classmethod
    def from_ppc(cls, ppc, sbase_mva=None):
        """Build a NetworkData from a PyPower/MATPOWER-format dict.

        Only in-service generators and branches are kept. Branches with
        RATE_A = 0 are treated as thermally unlimited: their limit is set to
        max(total in-service generation capacity, 1000 MW). Loads come from
        PD > 0 buses. sbase_mva defaults to ppc['baseMVA'] (100 MVA if the
        dict has no baseMVA).
        """
        bus_arr    = np.array(ppc['bus'],    dtype=float)
        gen_arr    = np.array(ppc['gen'],    dtype=float)
        branch_arr = np.array(ppc['branch'], dtype=float)

        if sbase_mva is None:
            sbase_mva = float(ppc.get('baseMVA', 100.0))

        # BUS_I ids (any numbering) -> 0-indexed row positions
        bus_ids = bus_arr[:, 0].astype(int)
        bus_map = {bid: i for i, bid in enumerate(bus_ids)}
        n_buses = len(bus_ids)

        # BUS_TYPE == 3 -> slack (the first such bus is used)
        slack_mask = bus_arr[:, 1] == 3
        if not slack_mask.any():
            raise ValueError("No slack bus (BUS_TYPE=3) found in ppc['bus'].")
        slack_bus = int(np.where(slack_mask)[0][0])

        # in-service only: GEN_STATUS col 7 == 1
        if gen_arr.shape[1] > 7:
            gen_mask = gen_arr[:, 7] == 1
        else:
            gen_mask = np.ones(gen_arr.shape[0], dtype=bool)
        gen_active = gen_arr[gen_mask]
        n_generators = gen_active.shape[0]

        gen_buses = [bus_map[int(g)] for g in gen_active[:, 0]]
        gen_pmin  = np.zeros(n_generators)               # Pmin ignored: no commitment variables
        gen_pmax  = gen_active[:, 8]                     # PMAX (col 8)

        # Cost = linear coefficient c_1 of a polynomial gencost row (MODEL 2);
        # any other row leaves the cost at 0. Without gencost every unit costs 1.
        gencost = ppc.get('gencost', None)
        if gencost is not None:
            gc = np.array(gencost, dtype=float)
            if gc.shape[0] == gen_arr.shape[0]:
                gc = gc[gen_mask]
            gen_cost = np.zeros(n_generators)
            for g in range(n_generators):
                model = int(gc[g, 0])
                ncost = int(gc[g, 3])
                if model == 2 and ncost >= 2:
                    # Polynomial: [c_{n-1}, ..., c_1, c_0] at cols [4, ..., 4+n-1]
                    # Linear coefficient c_1 is at col 4 + (ncost - 2)
                    li = 4 + ncost - 2
                    if li < gc.shape[1]:
                        gen_cost[g] = gc[g, li]
        else:
            gen_cost = np.ones(n_generators)

        # buses where PD > 0
        load_mask    = bus_arr[:, 2] > 1e-6
        load_buses   = [bus_map[int(bid)] for bid in bus_arr[load_mask, 0]]
        load_nominal = bus_arr[load_mask, 2]
        n_loads      = len(load_buses)

        if n_loads == 0:
            raise ValueError("No load buses found (PD > 0) in ppc['bus'].")

        # in-service only: BR_STATUS col 10 == 1
        if branch_arr.shape[1] > 10:
            br_mask = branch_arr[:, 10] == 1
        else:
            br_mask = np.ones(branch_arr.shape[0], dtype=bool)
        br_active = branch_arr[br_mask]
        n_lines   = br_active.shape[0]

        line_connections = [
            (bus_map[int(br_active[l, 0])], bus_map[int(br_active[l, 1])])
            for l in range(n_lines)
        ]

        # Susceptance in MW/rad = S_base / x_pu  (DC power flow convention)
        x_pu = br_active[:, 3]  # BR_X
        x_pu = np.where(np.abs(x_pu) < 1e-10, 1e-10, x_pu)
        line_susceptances = sbase_mva / x_pu

        # Flow limits in MW (RATE_A, col 5). RATE_A = 0 means unlimited; such
        # branches get max(total gen capacity, 1000 MW).
        rate_a          = br_active[:, 5]
        default_limit   = max(float(gen_pmax.sum()), 1000.0)
        line_flow_max   = np.where(rate_a > 1e-6, rate_a, default_limit)
        line_flow_min   = -line_flow_max

        # Bypass __init__, which would fill in the toy network.
        instance = cls.__new__(cls)
        instance.n_buses      = n_buses
        instance.n_lines      = n_lines
        instance.n_generators = n_generators
        instance.n_loads      = n_loads

        instance.gen_buses    = gen_buses
        instance.load_buses   = load_buses
        instance.slack_bus    = slack_bus

        instance.gen_pmin     = gen_pmin
        instance.gen_pmax     = gen_pmax
        instance.gen_cost     = gen_cost

        instance.load_nominal = load_nominal

        instance.line_connections  = line_connections
        instance.line_susceptances = line_susceptances
        instance.line_flow_min     = line_flow_min
        instance.line_flow_max     = line_flow_max

        instance.sbase_mva = sbase_mva

        instance._build_incidence_matrix()
        instance._build_susceptance_matrix()
        return instance


def create_network(case='toy', sbase_mva=100.0):
    """Return a NetworkData for the given case.

    'toy' returns the built-in 5-bus network. Any other string is looked up
    in the PGLib-OPF registry and downloaded on first use
    (~/.cache/pglib_opf/). A name that is not a registry key resolves by bus
    count ('case14' -> 'case14_ieee'), falling back to the closest bus count.
    For PGLib cases sbase_mva replaces the case's baseMVA, so it should equal
    the base of the case data (100 MVA for case14_ieee); 'toy' ignores it.
    """
    if case == 'toy':
        return NetworkData()
    return create_pglib_network(case, sbase_mva=sbase_mva)


def create_pglib_network(case_name, sbase_mva=None, cache_dir=None):
    """Download and parse a PGLib-OPF test case, returning a NetworkData.

    Accepts short keys like 'case14_ieee' or bare bus counts like 'case14'.
    A 'pglib_opf_' or 'pglib_' prefix is stripped. If sbase_mva is given it
    replaces the case's baseMVA. Requires pandapower.
    """
    try:
        import pandapower.converter as pc
    except ImportError:
        raise ImportError(
            "pandapower is required to load PGLib cases. "
            "Install with: pip install -r requirements.txt (pandapower==3.2.1, matpowercaseframes)"
        )

    case_key, exact = _resolve_pglib_case(case_name)

    if not exact:
        actual_n = _PGLIB_CASES[case_key][0]
        print(
            f"  Note: '{case_name}' is not in the PGLib registry. "
            f"Using closest match '{case_key}' ({actual_n} buses) instead."
        )

    local_path = _download_pglib_case(case_key, cache_dir=cache_dir)

    net = pc.from_mpc(str(local_path), validate_conversion=False)
    ppc = pc.to_ppc(net, init='flat', check_connectivity=False)

    if sbase_mva is not None:
        ppc['baseMVA'] = float(sbase_mva)

    return NetworkData.from_ppc(ppc)


# Alias of create_pglib_network.
create_ieee_network = create_pglib_network


if __name__ == "__main__":
    import sys
    case = sys.argv[1] if len(sys.argv) > 1 else 'toy'
    network = create_network(case)
    network.print_network_info()
