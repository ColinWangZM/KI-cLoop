from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("KGRAG_DATA_DIR", ROOT / "data")).expanduser()
OUTDIR = Path(__file__).resolve().parent
USPTO_PATH = Path(
    os.environ.get("KGRAG_REACTION_SOURCE", DATA_DIR / "private/reactions.csv")
).expanduser()
OR_PATH = Path(
    os.environ.get("KGRAG_OR_SUMMARY", DATA_DIR / "private/or_summary.csv")
).expanduser()
KINETICS_DESIGN_PATH = Path(
    os.environ.get(
        "KGRAG_KINETICS_DESIGN",
        DATA_DIR / "private/kinetics_outputs/route_experimental_design_summary.csv",
    )
).expanduser()
KINETICS_PARAM_PATH = Path(
    os.environ.get(
        "KGRAG_KINETICS_PARAMETERS",
        DATA_DIR / "private/kinetics_outputs/route_kinetic_parameter_summary.csv",
    )
).expanduser()


RNG = np.random.default_rng(42)
N_USPTO_ROWS = 100000
N_BACKGROUND = 12000
NBITS = 2048
X_PLOT_SCALE = 0.82
SPARSE_COLOR = "#1F3F73"
REACTION_TYPE_PALETTE = {
    "Peroxide chemistry": "#D77A61",
    "Oxidation/reduction": "#4C78A8",
    "Acylation/esterification": "#E69F43",
    "C-C coupling/aryl substitution": "#6BAE75",
    "C-N bond formation": "#9B7AA3",
    "C-O/S bond formation": "#63A6A3",
    "Halogen chemistry": "#C8AE4A",
    "Heterocycle chemistry": "#8B6F5C",
    "Other organic transformation": "#9A9A9A",
}

RDLogger.DisableLog("rdApp.*")


def strip_atom_mapping(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True)


def mol_fp(smiles: str, radius: int = 2, nbits: int = NBITS) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    arr[list(fp.GetOnBits())] = 1
    return arr


def reaction_fp(reactants: str, products: str, radius: int = 2, nbits: int = NBITS) -> np.ndarray | None:
    r_parts = [x for x in str(reactants).split(".") if x.strip()]
    p_parts = [x for x in str(products).split(".") if x.strip()]
    if not r_parts or not p_parts:
        return None
    r_fp = np.zeros(nbits, dtype=np.int16)
    p_fp = np.zeros(nbits, dtype=np.int16)
    for smi in r_parts:
        fp = mol_fp(smi, radius, nbits)
        if fp is None:
            return None
        r_fp += fp
    for smi in p_parts:
        fp = mol_fp(smi, radius, nbits)
        if fp is None:
            return None
        p_fp += fp
    return (p_fp - r_fp).astype(np.float32)


def contains_peroxide(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    pattern = Chem.MolFromSmarts("[O]-[O]")
    return bool(mol.HasSubstructMatch(pattern))


def has_smarts(smiles: str, smarts: str) -> bool:
    mol = Chem.MolFromSmiles(str(smiles))
    patt = Chem.MolFromSmarts(smarts)
    return bool(mol is not None and patt is not None and mol.HasSubstructMatch(patt))


def atom_count(smiles: str, atomic_num: int) -> int:
    total = 0
    for part in str(smiles).split("."):
        mol = Chem.MolFromSmiles(part)
        if mol is not None:
            total += sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == atomic_num)
    return total


def classify_reaction_type(reactants: str, products: str) -> str:
    joined = f"{reactants}.{products}"
    if contains_peroxide(joined):
        return "Peroxide chemistry"
    if atom_count(products, 7) > atom_count(reactants, 7):
        return "C-N bond formation"
    if atom_count(products, 8) + atom_count(products, 16) > atom_count(reactants, 8) + atom_count(reactants, 16):
        return "C-O/S bond formation"
    if any(has_smarts(joined, smarts) for smarts in ["C(=O)O", "C(=O)Cl", "C(=O)N"]):
        return "Acylation/esterification"
    if any(has_smarts(joined, smarts) for smarts in ["[Cl,Br,I,F]", "[C:1][Cl,Br,I]"]):
        return "Halogen chemistry"
    if any(has_smarts(joined, smarts) for smarts in ["[nH0]1cccc1", "n1ccccc1", "o1cccc1", "s1cccc1"]):
        return "Heterocycle chemistry"
    if "c1" in str(reactants) or "c1" in str(products):
        return "C-C coupling/aryl substitution"
    if atom_count(products, 8) != atom_count(reactants, 8):
        return "Oxidation/reduction"
    return "Other organic transformation"


def parse_operation_temperature(text: str) -> tuple[float | None, float | None]:
    values = [float(x) for x in re.findall(r"(-?\d+(?:\.\d+)?)\s*[°º]\s*C", str(text), flags=re.I)]
    if not values:
        values = [float(x) for x in re.findall(r"(-?\d+(?:\.\d+)?)\s*(?:degC|degrees? C|℃)", str(text), flags=re.I)]
    if not values:
        return None, None
    return min(values), max(values)


def parse_long_time_hours(text: str) -> float | None:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(hour|hours|h)\b", str(text), flags=re.I)
    if not matches:
        return None
    return max(float(value) for value, _ in matches)


def manifestation_flags(row: pd.Series) -> dict[str, bool]:
    joined = f"{row.get('reactants', '')}.{row.get('products', '')}"
    temp_min = row.get("temp_min")
    temp_max = row.get("temp_max")
    long_hours = row.get("long_time_h")
    catalyst_min = row.get("catalyst_min")
    catalyst_max = row.get("catalyst_max")
    raw_has_side = row.get("has_side_reaction", False)
    has_side = bool(raw_has_side) if pd.notna(raw_has_side) else False
    return {
        "hazardous_groups": bool(
            contains_peroxide(joined)
            or has_smarts(joined, "C(=O)Cl")
            or has_smarts(joined, "[N+](=O)[O-]")
            or has_smarts(joined, "[Cl,Br,I]")
        ),
        "extreme_conditions": bool(
            pd.notna(temp_max)
            and temp_max >= 80
            or pd.notna(temp_min)
            and temp_min <= 10
            or pd.notna(long_hours)
            and long_hours >= 12
        ),
        "operating_window": bool(
            pd.notna(temp_min)
            and pd.notna(temp_max)
            and temp_max - temp_min >= 20
            or pd.notna(catalyst_min)
            and pd.notna(catalyst_max)
            and catalyst_max > catalyst_min
        ),
        "mechanistic_uncertainty": has_side,
        "scale_up_limit": bool(str(row.get("context", "")).lower().find("kg/h") >= 0),
    }


def load_uspto_background() -> pd.DataFrame:
    df = pd.read_csv(USPTO_PATH, nrows=N_USPTO_ROWS, usecols=["Reactant", "Product", "PatentNumber", "Year"])
    df = df.dropna(subset=["Reactant", "Product"]).copy()
    df["reactants"] = df["Reactant"].map(strip_atom_mapping)
    df["products"] = df["Product"].map(strip_atom_mapping)
    df = df.dropna(subset=["reactants", "products"]).drop_duplicates(["reactants", "products"])
    if len(df) > N_BACKGROUND:
        df = df.sample(N_BACKGROUND, random_state=42)
    df["source"] = "USPTO"
    df["highlight"] = "background"
    df["reaction_smiles"] = df["reactants"] + ">>" + df["products"]
    df["context"] = ""
    df["temp_min"] = np.nan
    df["temp_max"] = np.nan
    df["long_time_h"] = np.nan
    df["has_side_reaction"] = False
    df["curated_route"] = ""
    return df


def load_kgrag_or_reactions(max_rows: int = 400) -> pd.DataFrame:
    if not OR_PATH.exists():
        return pd.DataFrame(columns=["source", "highlight", "reaction_smiles", "reactants", "products", "PatentNumber", "Year"])
    raw = pd.read_csv(OR_PATH)
    rows = []
    for _, row in raw.iterrows():
        context = str(row.get("node_context", ""))
        reactants = re.findall(r"反应物SMILES:\s*([^\n\r]*)", context)
        products = re.findall(r"生成物SMILES:\s*([^\n\r]*)", context)
        for reactant, product in zip(reactants, products):
            reactant = reactant.strip()
            product = product.strip()
            if not reactant or not product:
                continue
            r = strip_atom_mapping(reactant)
            p = strip_atom_mapping(product)
            if not r or not p:
                continue
            temp_min, temp_max = parse_operation_temperature(context)
            rows.append(
                {
                    "source": "KGRAG_OR",
                    "highlight": "peroxide_or_route" if contains_peroxide(r + "." + p) else "kgrag_or_route",
                    "reaction_smiles": f"{r}>>{p}",
                    "reactants": r,
                    "products": p,
                    "PatentNumber": "",
                    "Year": np.nan,
                    "context": context,
                    "temp_min": temp_min,
                    "temp_max": temp_max,
                    "long_time_h": parse_long_time_hours(context),
                    "has_side_reaction": bool(re.search(r"side|byproduct|副", context, flags=re.I)),
                    "curated_route": "",
                }
            )
    df = pd.DataFrame(rows).drop_duplicates("reaction_smiles")
    if len(df) > max_rows:
        peroxide = df[df["highlight"].eq("peroxide_or_route")]
        other = df[~df["highlight"].eq("peroxide_or_route")]
        other = other.sample(max(0, max_rows - len(peroxide)), random_state=42) if len(other) else other
        df = pd.concat([peroxide, other], ignore_index=True).head(max_rows)
    return df


def load_curated_kinetics_reactions() -> pd.DataFrame:
    design = pd.read_csv(KINETICS_DESIGN_PATH).set_index("route") if KINETICS_DESIGN_PATH.exists() else pd.DataFrame()
    params = pd.read_csv(KINETICS_PARAM_PATH).set_index("route") if KINETICS_PARAM_PATH.exists() else pd.DataFrame()
    curated = [
        {
            "route": "tbhp_bzcl",
            "source": "KGRAG_kinetics",
            "highlight": "curated_peroxide_kinetics",
            "reactants": "CC(C)(C)Cl.OO",
            "products": "CC(C)(C)OO",
            "reaction_smiles": "CC(C)(C)Cl.OO>>CC(C)(C)OO",
            "curated_route": "R1 TBHP-BZCL",
            "context": "Curated kinetics route TBHP-BZCL; t-BuCl + H2O2 -> TBHP.",
            "has_side_reaction": False,
        },
        {
            "route": "tbhp_wpo4",
            "source": "KGRAG_kinetics",
            "highlight": "curated_peroxide_kinetics",
            "reactants": "CC(C)(C)O.OO",
            "products": "CC(C)(C)OO.CC(C)(C)OOC(C)(C)C",
            "reaction_smiles": "CC(C)(C)O.OO>>CC(C)(C)OO.CC(C)(C)OOC(C)(C)C",
            "curated_route": "R2 TBHP-TBA-WPO4",
            "context": "Curated kinetics route TBHP-TBA-WPO4; TBA + H2O2 -> TBHP with DTBP side product.",
            "has_side_reaction": True,
        },
        {
            "route": "tbhp_cf3",
            "source": "KGRAG_kinetics",
            "highlight": "curated_peroxide_kinetics",
            "reactants": "CC(C)(C)O.OO",
            "products": "CC(C)(C)OO.CC(C)(C)OOC(C)(C)C",
            "reaction_smiles": "CC(C)(C)O.OO>>CC(C)(C)OO.CC(C)(C)OOC(C)(C)C",
            "curated_route": "R3 TBHP-CF3",
            "context": "Curated kinetics route TBHP-CF3; TBA + H2O2 -> TBHP with DTBP side product under CF3 acid catalysis.",
            "has_side_reaction": True,
        },
        {
            "route": "tbpb_benzaldehyde",
            "source": "KGRAG_kinetics",
            "highlight": "curated_peroxide_kinetics",
            "reactants": "O=CC1=CC=CC=C1.CC(C)(C)OO",
            "products": "CC(C)(C)OOC(=O)C1=CC=CC=C1",
            "reaction_smiles": "O=CC1=CC=CC=C1.CC(C)(C)OO>>CC(C)(C)OOC(=O)C1=CC=CC=C1",
            "curated_route": "R4 TBPB-benzaldehyde",
            "context": "Curated kinetics route TBPB-benzaldehyde; benzaldehyde + TBHP -> TBPB.",
            "has_side_reaction": False,
        },
    ]
    rows = []
    for item in curated:
        row = dict(item)
        route = row["route"]
        if not design.empty and route in design.index:
            for col in ["temp_min", "temp_max", "catalyst_min", "catalyst_max"]:
                row[col] = float(design.loc[route, col]) if pd.notna(design.loc[route, col]) else np.nan
        else:
            row.update({"temp_min": np.nan, "temp_max": np.nan, "catalyst_min": np.nan, "catalyst_max": np.nan})
        if not params.empty and route in params.index:
            row["Ea1_kJ_mol"] = float(params.loc[route, "Ea1_kJ_mol"])
            row["product_R2"] = float(params.loc[route, "product_R2"])
        row["long_time_h"] = np.nan
        row["PatentNumber"] = ""
        row["Year"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_embedding_table() -> pd.DataFrame:
    uspto = load_uspto_background()
    kgrag = load_kgrag_or_reactions()
    curated = load_curated_kinetics_reactions()
    df = pd.concat([uspto, kgrag, curated], ignore_index=True)
    fps = []
    keep = []
    for idx, row in df.iterrows():
        fp = reaction_fp(row["reactants"], row["products"])
        if fp is not None:
            keep.append(idx)
            fps.append(fp)
    df = df.loc[keep].reset_index(drop=True)
    x = np.vstack(fps)
    pca50 = PCA(n_components=min(50, x.shape[0] - 1), random_state=42).fit_transform(x)
    coords = TSNE(
        n_components=2,
        perplexity=min(35, max(5, (len(df) - 1) // 4)),
        init="pca",
        learning_rate="auto",
        random_state=42,
        n_iter=1200,
    ).fit_transform(pca50)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["reaction_type"] = [
        classify_reaction_type(r, p) for r, p in zip(df["reactants"], df["products"], strict=False)
    ]
    df["cluster"] = KMeans(n_clusters=9, random_state=42, n_init=20).fit_predict(pca50)

    nn = NearestNeighbors(n_neighbors=min(16, len(df))).fit(coords)
    distances, _ = nn.kneighbors(coords)
    df["local_sparsity"] = distances[:, -1]
    for key in ["hazardous_groups", "extreme_conditions", "operating_window", "mechanistic_uncertainty", "scale_up_limit"]:
        df[key] = [manifestation_flags(row)[key] for _, row in df.iterrows()]
    df["sparse_region"] = df["local_sparsity"] >= df["local_sparsity"].quantile(0.975)
    evidence_cols = ["hazardous_groups", "extreme_conditions", "operating_window", "mechanistic_uncertainty", "scale_up_limit"]
    df["evidence_linked_sparse_region"] = df["sparse_region"] & df[evidence_cols].fillna(False).any(axis=1)
    return df


def compute_sparse_circles(df: pd.DataFrame, max_circles: int = 5) -> pd.DataFrame:
    evidence_cols = ["hazardous_groups", "extreme_conditions", "operating_window", "mechanistic_uncertainty", "scale_up_limit"]
    evidence_mask = df[evidence_cols].fillna(False).any(axis=1)
    sparse = df[df["sparse_region"] & evidence_mask].copy()
    if len(sparse) < 5:
        return pd.DataFrame(columns=["sparse_cluster", "center_x", "center_y", "radius", "n_points"])
    n_clusters = min(max_circles, len(sparse))
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(sparse[["x", "y"]])
    sparse["sparse_cluster"] = labels
    circles = []
    for cluster_id, sub in sparse.groupby("sparse_cluster"):
        if len(sub) < 3:
            continue
        cx, cy = sub[["x", "y"]].mean()
        radius = np.sqrt(((sub["x"] - cx) ** 2 + (sub["y"] - cy) ** 2).quantile(0.55)) * 1.05
        circles.append(
            {
                "sparse_cluster": int(cluster_id),
                "center_x": float(cx),
                "center_y": float(cy),
                "plot_center_x": float(cx * X_PLOT_SCALE),
                "plot_center_y": float(cy),
                "radius": float(min(max(radius, 1.5), 7.5)),
                "plot_radius": float(min(max(radius, 1.5), 7.5) * 0.88),
                "n_points": int(len(sub)),
                "mean_local_sparsity": float(sub["local_sparsity"].mean()),
                "dominant_reaction_type": str(sub["reaction_type"].mode().iloc[0]) if len(sub) else "",
            }
        )
    return pd.DataFrame(circles)


def plot(df: pd.DataFrame, circles: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(6.7, 5.2), facecolor="none")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.48], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("none")
    legend_ax.set_facecolor("none")
    legend_ax.set_axis_off()
    visible = df[~df["highlight"].eq("curated_peroxide_kinetics")]
    for reaction_type, sub in visible.groupby("reaction_type"):
        ax.scatter(
            sub["x"] * X_PLOT_SCALE,
            sub["y"],
            s=4.9,
            color=REACTION_TYPE_PALETTE.get(reaction_type, "#BAB0AC"),
            alpha=0.68,
            linewidths=0,
            label=reaction_type,
        )

    for _, circle in circles.iterrows():
        ax.add_patch(
            Circle(
                (circle["plot_center_x"], circle["plot_center_y"]),
                circle["plot_radius"],
                fill=False,
                lw=1.6,
                ls="-",
                ec=SPARSE_COLOR,
                alpha=0.95,
            )
        )

    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmid = (xlim[0] + xlim[1]) / 2
    ymid = (ylim[0] + ylim[1]) / 2
    half = max((xlim[1] - xlim[0]), (ylim[1] - ylim[0])) / 2
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    type_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=4.6, label=label)
        for label, color in REACTION_TYPE_PALETTE.items()
    ]
    sparse_handle = [Line2D([0], [0], color=SPARSE_COLOR, lw=1.6, ls="-", label="Sparse regions")]
    legend_top = legend_ax.legend(
        handles=type_handles[:5],
        loc="upper left",
        bbox_to_anchor=(-0.20, 0.82),
        frameon=False,
        fontsize=6.8,
        handletextpad=0.34,
        labelspacing=0.32,
        borderaxespad=0,
    )
    legend_ax.add_artist(legend_top)
    legend_ax.legend(
        handles=type_handles[5:] + sparse_handle,
        loc="lower left",
        bbox_to_anchor=(-0.20, 0.18),
        frameon=False,
        fontsize=6.8,
        handletextpad=0.34,
        labelspacing=0.32,
        borderaxespad=0,
        ncol=1,
    )
    fig.savefig(OUTDIR / "reaction_space_sparse_map.png", dpi=450, bbox_inches="tight", transparent=True)
    fig.savefig(OUTDIR / "reaction_space_sparse_map.pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)


def write_sparse_statistics(df: pd.DataFrame) -> None:
    evidence_cols = ["hazardous_groups", "extreme_conditions", "operating_window", "mechanistic_uncertainty", "scale_up_limit"]
    df["evidence_linked_sparse_region"] = df["sparse_region"] & df[evidence_cols].fillna(False).any(axis=1)
    summary_rows = []
    for group_name, sub in [
        ("all_points", df),
        ("sparse_region_points", df[df["sparse_region"]]),
        ("evidence_linked_sparse_region_points", df[df["evidence_linked_sparse_region"]]),
        ("kgrag_kinetics_routes", df[df["highlight"].eq("curated_peroxide_kinetics")]),
    ]:
        row = {"group": group_name, "n": len(sub)}
        for col in evidence_cols:
            row[f"{col}_n"] = int(sub[col].fillna(False).sum()) if len(sub) else 0
            row[f"{col}_fraction"] = float(sub[col].fillna(False).mean()) if len(sub) else 0.0
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "sparse_region_manifestation_statistics.csv", index=False)

    curated_cols = [
        "curated_route",
        "reaction_smiles",
        "reaction_type",
        "local_sparsity",
        "sparse_region",
        "evidence_linked_sparse_region",
        "hazardous_groups",
        "extreme_conditions",
        "operating_window",
        "mechanistic_uncertainty",
        "scale_up_limit",
        "temp_min",
        "temp_max",
        "catalyst_min",
        "catalyst_max",
        "Ea1_kJ_mol",
        "product_R2",
        "context",
    ]
    df[df["highlight"].eq("curated_peroxide_kinetics")][curated_cols].to_csv(
        OUTDIR / "kgrag_peroxide_sparse_evidence.csv", index=False
    )


def write_reproducibility_outputs(df: pd.DataFrame, circles: pd.DataFrame) -> None:
    evidence_cols = ["hazardous_groups", "extreme_conditions", "operating_window", "mechanistic_uncertainty", "scale_up_limit"]
    input_summary = (
        df.groupby(["source", "highlight"], dropna=False)
        .size()
        .reset_index(name="n_reactions_after_fingerprint_filter")
        .sort_values(["source", "highlight"])
    )
    input_summary.to_csv(OUTDIR / "reaction_space_input_summary.csv", index=False)

    type_summary = (
        df.groupby("reaction_type", dropna=False)
        .agg(
            n=("reaction_smiles", "size"),
            sparse_region_n=("sparse_region", "sum"),
            evidence_linked_sparse_region_n=("evidence_linked_sparse_region", "sum"),
            mean_local_sparsity=("local_sparsity", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    type_summary.to_csv(OUTDIR / "reaction_type_statistics.csv", index=False)

    sparse_cols = [
        "source",
        "highlight",
        "reaction_smiles",
        "reactants",
        "products",
        "reaction_type",
        "x",
        "y",
        "local_sparsity",
        "sparse_region",
        "evidence_linked_sparse_region",
        *evidence_cols,
        "context",
    ]
    df[df["sparse_region"]][[col for col in sparse_cols if col in df.columns]].to_csv(
        OUTDIR / "reaction_space_sparse_points.csv", index=False
    )
    circles.to_csv(OUTDIR / "reaction_space_sparse_circles.csv", index=False)

    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": 42,
        "input_paths": {
            "uspto_background": str(USPTO_PATH),
            "kgrag_or_contexts": str(OR_PATH),
            "kinetics_design_summary": str(KINETICS_DESIGN_PATH),
            "kinetics_parameter_summary": str(KINETICS_PARAM_PATH),
        },
        "sampling": {
            "n_uspto_rows_read": N_USPTO_ROWS,
            "n_background_sampled": N_BACKGROUND,
            "kgrag_or_max_rows": 400,
        },
        "fingerprint": {
            "type": "Morgan reaction-difference fingerprint",
            "radius": 2,
            "n_bits": NBITS,
            "formula": "sum(product Morgan fingerprints) - sum(reactant Morgan fingerprints)",
        },
        "embedding": {
            "pca_n_components": min(50, len(df) - 1),
            "tsne_n_components": 2,
            "tsne_perplexity": min(35, max(5, (len(df) - 1) // 4)),
            "tsne_n_iter": 1200,
            "tsne_init": "pca",
            "tsne_learning_rate": "auto",
            "kmeans_n_clusters": 9,
            "kmeans_n_init": 20,
        },
        "sparse_region_definition": {
            "nearest_neighbor_rank": 15,
            "local_sparsity_metric": "distance to 15th nearest neighbor in 2D embedding",
            "sparse_quantile": 0.975,
            "evidence_linked_rule": "sparse_region AND any key manifestation flag",
            "max_sparse_circles": 5,
        },
        "manifestation_columns": evidence_cols,
        "reaction_type_palette": REACTION_TYPE_PALETTE,
        "plot": {
            "x_plot_scale": X_PLOT_SCALE,
            "sparse_circle_color": SPARSE_COLOR,
            "scatter_point_size": 4.9,
        },
        "output_files": [
            "reaction_space_sparse_map.png",
            "reaction_space_sparse_map.pdf",
            "reaction_space_sparse_map_points.csv",
            "reaction_space_sparse_points.csv",
            "reaction_space_sparse_circles.csv",
            "reaction_space_input_summary.csv",
            "reaction_type_statistics.csv",
            "sparse_region_manifestation_statistics.csv",
            "kgrag_peroxide_sparse_evidence.csv",
            "reaction_space_reproducibility_config.json",
        ],
    }
    with (OUTDIR / "reaction_space_reproducibility_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = build_embedding_table()
    circles = compute_sparse_circles(df)
    df.to_csv(OUTDIR / "reaction_space_sparse_map_points.csv", index=False)
    plot(df, circles)
    write_sparse_statistics(df)
    write_reproducibility_outputs(df, circles)
    provenance = f"""# Reaction-space sparse map provenance

This figure is a local, reproducible approximation inspired by RXNFP reaction atlases.

## Inputs

- USPTO background reactions: `{USPTO_PATH}`
- KGRAG organic reaction contexts: `{OR_PATH}`
- Curated kinetics routes: TBHP-BZCL, TBHP-TBA-WPO4, TBHP-CF3, and TBPB-benzaldehyde
- Route design statistics: `{KINETICS_DESIGN_PATH}`
- Route kinetic parameters: `{KINETICS_PARAM_PATH}`

## Method

- Reaction SMILES are constructed as `reactants>>products`.
- Atom-map numbers are stripped with RDKit before fingerprinting.
- Reaction fingerprint: Morgan(product sum) - Morgan(reactant sum), radius 2, {NBITS} bits.
- Dimensionality reduction: PCA to 50 dimensions followed by t-SNE to 2 dimensions.
- Background colors: rule-based reaction type classes inferred from reactant/product SMILES, including peroxide chemistry, oxidation/reduction, acylation/esterification, C-C/aromatic substitution, C-N formation, C-O/S formation, halogen chemistry, heterocycle chemistry, and other transformations.
- Sparse regions: top 2.5% by 15-nearest-neighbour distance in the 2D embedding.
- Evidence-linked sparse regions: sparse-region points that also carry at least one key manifestation flag, grouped and circled automatically.
- Sparse-region evidence follows the paper logic:
  - `hazardous_groups`: peroxide O-O, acyl chloride, nitro, or halogen-containing hazards.
  - `extreme_conditions`: temperature >=80 °C, temperature <=10 °C, or operation time >=12 h when available from source text or kinetics design.
  - `operating_window`: route-level temperature window >=20 °C or scanned catalyst window from measured experiments.
  - `mechanistic_uncertainty`: explicit side/byproduct information; in the four kinetics routes this captures DTBP side formation for the two TBA-to-TBHP routes.
  - `scale_up_limit`: source text reports kg/h-scale operation.

The four KGRAG peroxide kinetic routes are included in the data table for evidence/statistical linkage, but are not explicitly labelled or marked in the plotted panel.

## Outputs

- `reaction_space_sparse_map.png`
- `reaction_space_sparse_map.pdf`
- `reaction_space_sparse_map_points.csv`
- `reaction_space_sparse_points.csv`
- `reaction_space_sparse_circles.csv`
- `reaction_space_input_summary.csv`
- `reaction_type_statistics.csv`
- `sparse_region_manifestation_statistics.csv`
- `kgrag_peroxide_sparse_evidence.csv`
- `reaction_space_reproducibility_config.json`
"""
    (OUTDIR / "README.md").write_text(provenance)
    print(OUTDIR / "reaction_space_sparse_map.png")


if __name__ == "__main__":
    main()
