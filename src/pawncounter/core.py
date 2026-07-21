"""pawncounter: pawn-structure state-graph exploration helpers.

The board size (and where generated data lives) are project-global parameters, so the
reusable, *non-generation* helpers are gathered as methods on `PawnCounter`, constructed
once per board and used qualified:

    import pawncounter

    pc = pawncounter.PawnCounter(2, 3)
    pc.position_chart(pc.init_position())

Generation (the move rules `accessible_positions` and the BFS `explore_transitions`) also
lives here. `generate_transitions()` is the lazy entry point that ties everything
together: it loads `pc.data_dir/transitions.parquet` (the `transitions_WxD.tmp` folder)
if present, otherwise runs the BFS and caches it -- so notebook 01 just loops over board
sizes calling it, and notebook 02 calls it to get the same data back.
"""

from typing import Callable
from pathlib import Path
import functools

import numpy as np
import polars as pl
import altair as alt
import scipy.sparse as sp
from sklearn.manifold import spectral_embedding
import polyscope as ps
import ipywidgets as widgets


type Position = tuple[np.uint64, np.uint64]
"""Pawn structure position --- a configuration of White and Black pawns.

Two u64 bitmasks, the first for White pawns and the second for Black. The least bit of
each mask is the bottom-left square (a1), counting up in file-major order (a1, a2, ...,
a8, b1, ...) with a fixed stride of 8 squares per file, so the packing does not depend on
the board size --- a smaller board simply leaves the unused squares empty.
"""


class PawnCounter:
    """Board-parameterised bundle of pawncounter's non-generation helpers.

    `width` x `depth` is the board. Positions are u64 bitmasks with a fixed 8-per-file
    stride, so only the occupied squares depend on the board size. `data_dir` is where
    `cached_frame` reads/writes parquet (default `./transitions_{width}x{depth}.tmp`) ---
    the folder that connects the generation and exploration notebooks.
    """

    def __init__(self, width: int, depth: int, *, data_dir: str | Path | None = None):
        assert 0 <= width <= 8
        assert 0 <= depth <= 8
        self.width = width
        self.depth = depth
        self.data_dir = (
            Path(data_dir)
            if data_dir is not None
            else Path(f"./data/transitions_{width}x{depth}.tmp")
        )

    # ---- data caching ---------------------------------------------------

    def cached_frame(self, name: str):
        """Decorator: memoise a DataFrame builder to `{data_dir}/{name}.parquet`."""

        def cached_frame_inner(func: Callable[..., pl.DataFrame]):
            @functools.wraps(func)
            def inner_func(*args, **kwargs):
                self.data_dir.mkdir(parents=True, exist_ok=True)

                cached_frame_path = self.data_dir / f"{name}.parquet"
                if cached_frame_path.exists():
                    return pl.read_parquet(cached_frame_path)

                frame = func(*args, **kwargs)
                frame.write_parquet(cached_frame_path)
                return frame

            return inner_func

        return cached_frame_inner

    # ---- position representation ----------------------------------------

    def position_to_int(self, pos: Position) -> int:
        """Pack a Position into a single UInt128: low 64 bits White, high 64 bits Black."""
        white, black = pos
        return int(white) | (int(black) << 64)

    def position_from_int(self, code: int) -> Position:
        """Unpack a UInt128 (low bits White, high bits Black) back into a Position."""
        mask = (1 << 64) - 1
        return np.uint64(code & mask), np.uint64((code >> 64) & mask)

    def position_ndarray(self, pos: Position) -> tuple[np.ndarray, np.ndarray]:
        def to_array(mask):
            arr = np.zeros((self.width, self.depth), dtype=bool)
            for f in range(self.width):
                for r in range(self.depth):
                    arr[f, r] = (mask >> (f * 8 + r)) & 1
            return arr

        white, black = pos
        return to_array(white), to_array(black)

    def init_position(self) -> Position:
        white = 0
        black = 0
        for f in range(self.width):
            white |= 1 << (f * 8)
            black |= 1 << (f * 8 + self.depth - 1)
        return np.uint64(white), np.uint64(black)

    def rand_position(self, max_pawns_per_side: int | None = None) -> Position:
        squares = [f * 8 + r for f in range(self.width) for r in range(self.depth)]
        n = len(squares)
        cap = n if max_pawns_per_side is None else min(max_pawns_per_side, n)

        n_white = np.random.randint(cap + 1)
        n_black = np.random.randint(min(cap, n - n_white) + 1)

        chosen = np.random.choice(n, size=n_white + n_black, replace=False)
        white = 0
        black = 0
        for i in chosen[:n_white]:
            white |= 1 << squares[i]
        for i in chosen[n_white:]:
            black |= 1 << squares[i]
        return np.uint64(white), np.uint64(black)

    def pawns_as_frame(self, pos: Position) -> pl.DataFrame:
        colour = pl.Enum(["White", "Black"])
        rows = []
        for name, mask in zip(["White", "Black"], pos):
            for f in range(self.width):
                for r in range(self.depth):
                    if (mask >> (f * 8 + r)) & 1:
                        rows.append({"rank": r + 1, "file": f + 1, "colour": name})
        return pl.DataFrame(
            rows, schema={"rank": pl.Int64, "file": pl.Int64, "colour": colour}
        )

    # ---- transition generation ------------------------------------------

    def _on_board(self, f: int, r: int) -> bool:
        return 0 <= f < self.width and 0 <= r < self.depth

    def accessible_positions(self, pos: Position) -> list[tuple[str, int, Position]]:
        white, black = int(pos[0]), int(pos[1])
        out: list[tuple[str, int, Position]] = []

        def emit(kind, src, new_own, new_enemy, white_to_move):
            w, b = (new_own, new_enemy) if white_to_move else (new_enemy, new_own)
            out.append((kind, src, (np.uint64(w), np.uint64(b))))

        for white_to_move in (True, False):
            own, enemy = (white, black) if white_to_move else (black, white)
            dr = 1 if white_to_move else -1
            for f in range(self.width):
                for r in range(self.depth):
                    src = f * 8 + r
                    if not (own >> src) & 1:
                        continue
                    own_removed = own & ~(1 << src)

                    af, ar = f, r + dr  # advance (straight)
                    if self._on_board(af, ar):
                        dst = af * 8 + ar
                        if not ((own | enemy) >> dst) & 1:
                            emit("A", src, own_removed | (1 << dst), enemy, white_to_move)
                    else:
                        emit(
                            "A", src, own_removed, enemy, white_to_move
                        )  # off end rank == R

                    for kind, df in (("CQ", -1), ("CK", 1)):  # diagonal captures
                        cf, cr = f + df, r + dr
                        if not 0 <= cf < self.width:
                            continue  # off the side: not a legal capture
                        if not 0 <= cr < self.depth:
                            emit(
                                kind, src, own_removed, enemy, white_to_move
                            )  # off end rank == R
                            continue
                        dst = cf * 8 + cr
                        if (own >> dst) & 1:
                            continue  # friendly block
                        emit(
                            kind,
                            src,
                            own_removed | (1 << dst),
                            enemy & ~(1 << dst),
                            white_to_move,
                        )

                    emit("R", src, own_removed, enemy, white_to_move)  # always

        return out

    def explore_transitions(self, start: Position) -> pl.DataFrame:
        """Every transition reachable from `start`, one row per transition.

        Positions are packed into a UInt128: low 64 bits White, high 64 bits Black.
        `transition_depth` is the minimum number of steps to reach the position the
        move is made from (the BFS level of `start_pos`).

        Edges are collected one BFS level at a time and concatenated, keeping peak
        memory bounded (~2.4 GB / ~2 min for the 4x4 start position: ~2.2M positions,
        ~46M transitions).
        """
        transition_type = pl.Enum(["A", "CQ", "CK", "R"])
        schema = {
            "start_pos": pl.UInt128,
            "end_pos": pl.UInt128,
            "moving_pawn": pl.UInt8,
            "transition_type": transition_type,
        }

        def encode(pos: Position) -> int:
            return int(pos[0]) | (int(pos[1]) << 64)

        seen = {encode(start)}
        frontier = [start]
        chunks: list[pl.DataFrame] = []

        depth = 0
        while frontier:
            starts, ends, pawns, kinds = [], [], [], []
            next_frontier = []
            for pos in frontier:
                start_code = encode(pos)
                for kind, src, end in self.accessible_positions(pos):
                    end_code = encode(end)
                    starts.append(start_code)
                    ends.append(end_code)
                    pawns.append(src)
                    kinds.append(kind)
                    if end_code not in seen:
                        seen.add(end_code)
                        next_frontier.append(end)
            if starts:
                chunks.append(
                    pl.DataFrame(
                        {
                            "start_pos": starts,
                            "end_pos": ends,
                            "moving_pawn": pawns,
                            "transition_type": kinds,
                        },
                        schema=schema,
                    ).with_columns(transition_depth=pl.lit(depth, dtype=pl.UInt8))
                )
            frontier = next_frontier
            depth += 1

        return pl.concat(chunks, rechunk=False)

    def generate_transitions(self) -> pl.DataFrame:
        """This board's transition table, generated lazily.

        Loads `{data_dir}/transitions.parquet` if it exists; otherwise runs the BFS from
        `init_position()`, caches the result there, and returns it. This is the single
        entry point that connects generation (notebook 01) with exploration (notebook 02).
        """

        @self.cached_frame("transitions")
        def build() -> pl.DataFrame:
            return self.explore_transitions(self.init_position())

        return build()

    # ---- position charts ------------------------------------------------

    def position_chart(
        self, pos: Position, *, axis=False, legend=False
    ) -> alt.LayerChart:
        _PAWN_COLOURS = {
            "domain": ["White", "Black"],
            "range": ["white", "black"],
        }
        _SQUARE_COLOURS = {
            "domain": ["light", "dark"],
            "range": ["#f0d9b5", "#b58863"],
        }
        _FILE_DOMAIN = list(range(1, self.width + 1))
        _RANK_DOMAIN = list(range(1, self.depth + 1))

        def _board_as_frame() -> pl.DataFrame:
            rows = []
            for f in range(1, self.width + 1):
                for r in range(1, self.depth + 1):
                    square = "dark" if (f + r) % 2 == 0 else "light"
                    rows.append({"rank": r, "file": f, "square": square})
            return pl.DataFrame(rows)

        axis_params = {} if axis else {"axis": None}
        legend_params = {} if legend else {"legend": None}

        board = (
            alt.Chart(_board_as_frame())
            .mark_rect()
            .encode(
                alt.X("file:O", **axis_params).scale(domain=_FILE_DOMAIN),  # type: ignore
                alt.Y("rank:O", **axis_params).scale(domain=_RANK_DOMAIN),  # type: ignore
                alt.Color("square:N")  # type: ignore
                .scale(**_SQUARE_COLOURS)  # type: ignore
                .legend(None),
            )
        )

        pawns = (
            alt.Chart(self.pawns_as_frame(pos))
            .mark_circle(size=250, stroke="black", strokeWidth=0.5)
            .encode(
                alt.X("file:O").scale(domain=_FILE_DOMAIN),
                alt.Y("rank:O").scale(domain=_RANK_DOMAIN, reverse=True),
                alt.Color("colour:N", **legend_params)  # type: ignore
                #
                .scale(**_PAWN_COLOURS),  # type: ignore
            )
        )

        return (
            alt.layer(board, pawns)
            .resolve_scale(color="independent")
            .properties(width=33 * self.width, height=33 * self.depth)
        )  # type: ignore

    def multiple_positions_chart(
        self, poses: list[Position], *, width: int | None = None
    ) -> alt.VConcatChart:
        pos_charts = [self.position_chart(p) for p in poses]
        if width is None:
            width = int(np.sqrt(len(pos_charts)))
        return alt.vconcat(
            *(
                alt.hconcat(*pos_charts[n : (n + width)])
                for n in range(0, len(pos_charts), width)
            )
        )

    # ---- position graph & spectral embedding ----------------------------

    def extract_positions(self, transitions: pl.DataFrame) -> pl.DataFrame:
        return (
            pl.concat(
                [
                    transitions.lazy().select(pos="start_pos"),
                    transitions.lazy().select(pos="end_pos"),
                ]
            )
            .select(pl.col("pos").unique(maintain_order=True))
            .with_row_index("position_id")
            .collect()
        )

    def positions_adjacency(
        self,
        transitions: pl.DataFrame,
        positions: pl.DataFrame,
        *,
        weight: pl.Expr = pl.lit(1.0),
    ) -> sp.csr_matrix:
        """Symmetric weighted adjacency of the position graph, indexed by `positions` row order.

        `weight` is a Polars expression evaluated against `transitions`, giving a weight per
        transition. Parallel transitions between the same pair are collapsed by taking the max
        weight; transitions are directed, so we symmetrise (undirected) by taking the max weight
        over the two directions, and drop self-loops. The default `weight=pl.lit(1.0)` reproduces
        the plain 0/1 adjacency.
        """
        nodes = positions
        edges = (
            transitions.select("start_pos", "end_pos", w=weight)
            .join(nodes.rename({"pos": "start_pos", "position_id": "i"}), on="start_pos")
            .join(nodes.rename({"pos": "end_pos", "position_id": "j"}), on="end_pos")
            .group_by("i", "j")
            .agg(pl.col("w").max())
        )
        i = edges["i"].to_numpy()
        j = edges["j"].to_numpy()
        w = edges["w"].to_numpy().astype(float)
        n = positions.height
        A = sp.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
        A = A.maximum(A.T)  # undirected: max weight over the two directions
        A.setdiag(0)  # no self-loops
        A.eliminate_zeros()
        return A  # type: ignore

    def spectral_embedding_3d(
        self,
        transitions: pl.DataFrame,
        positions: pl.DataFrame,
        *,
        n_dims: int = 3,
        weight: pl.Expr = pl.lit(1.0),
        eigen_solver: str = "amg",
    ) -> pl.DataFrame:
        """Spectral embedding of the position graph: `positions` plus (x, y, z) columns.

        `n_dims` (1..3) is how many embedding coordinates to actually compute; the unused axes
        among (x, y, z) are filled with 0 so the output schema never changes. `weight` is a
        Polars expression over `transitions` giving each edge's weight (default `1.0`, unweighted).

        Rows line up with `positions`. Coordinates are the smallest non-trivial eigenvectors
        of the normalised graph Laplacian. `eigen_solver="amg"` (pyamg-preconditioned LOBPCG)
        scales to millions of nodes; "arpack" (shift-invert) does NOT here -- fill-in on this
        expander-like graph makes it blow past minutes even at 3x4. "lobpcg" is fine small.
        """
        if not 1 <= n_dims <= 3:
            raise ValueError(f"n_dims must be between 1 and 3, got {n_dims}")
        A = self.positions_adjacency(transitions, positions, weight=weight)
        emb = spectral_embedding(
            A,
            n_components=n_dims,
            eigen_solver=eigen_solver,  # type: ignore
            drop_first=True,
            random_state=0,
        )
        n = positions.height
        coords = [emb[:, k] if k < n_dims else np.zeros(n) for k in range(3)]
        return positions.with_columns(
            x=pl.Series(coords[0]), y=pl.Series(coords[1]), z=pl.Series(coords[2])
        )

    # ---- rendering ------------------------------------------------------

    def button(self, description: str = "Run"):
        """Defer a side-effecting `func` (which returns None) behind an ipywidgets button.

        Use as a decorator, then call the wrapped function to get a Button instead of running
        `func`. Each click runs `func(*args, **kwargs)` with the args passed when the button was
        made. Display the button as the cell's last expression.
        """

        def button_inner(func: Callable[..., None]):
            @functools.wraps(func)
            def inner_func(*args, **kwargs):
                btn = widgets.Button(description=description)
                btn.on_click(lambda _clicked: func(*args, **kwargs))
                return btn

            return inner_func

        return button_inner

    def display_coloured_mesh(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        edge_colours: np.ndarray,
        *,
        name: str = "mesh",
        radius: float = 0.001,
        drop_unused_nodes: bool = True,
        normalise: bool = True,
        highlight_nodes: np.ndarray | None = None,
    ) -> ps.CurveNetwork:
        """Display `edges` as a polyscope line mesh, one RGB colour per edge.

        nodes:        (N, 3) float node positions.
        edges:        (M, 2) int index pairs into `nodes`.
        edge_colours: (M, 3) float RGB in [0, 1], row-aligned with `edges`.

        `drop_unused_nodes` reindexes onto only the nodes the edges touch, so the bounding
        box stays tight around what is actually drawn. `normalise` recentres and rescales
        each axis independently so the mesh fills a unit cube -- handy when coordinates are
        tiny or very anisotropic (as the spectral embedding's are); pass False to keep the
        true aspect ratio. `highlight_nodes` are node indices (into `nodes`) to mark with
        spheres; they get the same reindex/normalise transform so they line up with the mesh
        (indices dropped by `drop_unused_nodes` are skipped).
        """
        nodes = np.asarray(nodes, dtype=float)
        edges = np.asarray(edges)
        highlight = (
            None if highlight_nodes is None else np.asarray(highlight_nodes).reshape(-1)
        )

        if drop_unused_nodes:
            used = np.unique(edges)
            remap = np.full(len(nodes), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            edges = remap[edges]
            nodes = nodes[used]
            if highlight is not None:
                highlight = remap[highlight]
                highlight = highlight[highlight >= 0]

        if normalise:
            nodes = nodes - nodes.mean(axis=0)
            span = np.abs(nodes).max(axis=0)
            span[span == 0] = 1.0
            nodes = nodes / span

        print(
            f"{len(edges)} edges, {len(nodes)} nodes; extents "
            f"{nodes.min(axis=0).round(3)} .. {nodes.max(axis=0).round(3)}"
        )

        ps.init()
        net = ps.register_curve_network(name, nodes, edges)
        net.set_radius(radius, relative=True)  # visible tube; bump if still too thin
        net.add_color_quantity(
            "colour", np.asarray(edge_colours), defined_on="edges", enabled=True
        )

        if highlight is not None and len(highlight):
            marker = ps.register_point_cloud(f"{name} (initial state)", nodes[highlight])
            marker.set_radius(0.01, relative=True)
            marker.set_color((1.0, 0.85, 0.0))

        ps.show()
        return net
