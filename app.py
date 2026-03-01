"""Interactive web interface for the protein folding prototype.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from protein_folding import (
    ModelConfig,
    RealStructureDataset,
    TrainableProteinModel,
    save_multiview_image,
    save_rotation_gif,
    save_structure_image,
)

st.set_page_config(page_title="Protein Folding Studio", page_icon="🧬", layout="wide")


def _init_state() -> None:
    if "model" not in st.session_state:
        st.session_state.model = TrainableProteinModel(ModelConfig())
        st.session_state.model_loaded = False
        st.session_state.last_prediction = None
        st.session_state.last_sequence = ""


def _render_header() -> None:
    st.title("🧬 Protein Folding Studio")
    st.caption(
        "Train, predict, and visualize structures with graph/recycling trunks and optional e3nn SE(3)-equivariant updates."
    )


def _render_model_controls() -> None:
    with st.sidebar:
        st.header("Model setup")
        epochs = st.slider("Epochs", 1, 100, 15)
        batch_size = st.slider("Batch size", 1, 16, 2)
        lr = st.number_input("Learning rate", 1e-5, 1e-1, 5e-3, format="%.5f")
        recycles = st.slider("Recycles", 1, 8, 4)
        knn_k = st.slider("kNN neighbors", 2, 32, 8)
        use_radius_graph = st.checkbox("Use radius graph", value=False)
        radius_cutoff = st.slider("Radius cutoff", 4.0, 20.0, 10.0)
        use_e3nn = st.checkbox("Enable e3nn (if installed)", value=True)

        if st.button("Initialize model", type="primary"):
            cfg = ModelConfig(
                epochs=epochs,
                batch_size=batch_size,
                lr=float(lr),
                n_recycles=recycles,
                knn_k=knn_k,
                use_radius_graph=use_radius_graph,
                radius_cutoff=radius_cutoff,
                use_e3nn=use_e3nn,
            )
            st.session_state.model = TrainableProteinModel(cfg)
            st.session_state.model_loaded = True
            st.success("Model initialized.")

        st.markdown("---")
        st.subheader("Checkpoint")
        up = st.file_uploader("Load JSON checkpoint", type=["json"], accept_multiple_files=False)
        if up is not None and st.button("Load checkpoint"):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "weights.json"
                path.write_bytes(up.read())
                st.session_state.model.load(str(path))
            st.session_state.model_loaded = True
            st.success("Checkpoint loaded.")


def _train_tab() -> None:
    st.subheader("Training")
    st.write("Upload one or more local PDB files and run fitting.")
    files = st.file_uploader("Training PDB files", type=["pdb"], accept_multiple_files=True)

    if st.button("Train on uploaded PDBs"):
        if not files:
            st.warning("Please upload at least one PDB file.")
            return

        with tempfile.TemporaryDirectory() as td:
            paths = []
            for f in files:
                p = Path(td) / f.name
                p.write_bytes(f.read())
                paths.append(str(p))

            dataset = RealStructureDataset.from_local_pdbs(paths, min_len=5)
            if not dataset.examples:
                st.error("No valid structures parsed from the uploaded PDB files.")
                return

            with st.spinner("Training model..."):
                result = st.session_state.model.fit(dataset)
                metrics = st.session_state.model.evaluate(dataset)

        st.success("Training complete.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Epochs ran", result.epochs_ran)
        c2.metric("Best val loss", f"{result.best_val_loss:.4f}")
        c3.metric("Best epoch", result.best_epoch)
        st.json(metrics)


def _infer_tab() -> None:
    st.subheader("Prediction")
    sequence = st.text_input("Amino-acid sequence", value="ACDEFGHIKLMNPQ", help="Use one-letter amino-acid codes.")

    col1, col2 = st.columns([2, 1])
    with col1:
        predict_btn = st.button("Predict structure", type="primary")
    with col2:
        ensemble_n = st.number_input("Ensemble size", min_value=1, max_value=32, value=6, step=1)

    if predict_btn:
        pred = st.session_state.model.predict(sequence)
        st.session_state.last_prediction = pred
        st.session_state.last_sequence = sequence

    pred = st.session_state.get("last_prediction")
    if pred is None:
        st.info("Run prediction to see outputs.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Residues", len(pred.sequence))
    c2.metric("Mean confidence", f"{float(pred.confidence.mean()):.2f}")
    c3.metric("e3nn active", "Yes" if st.session_state.model._e3nn_ready else "No")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        img_path = td_path / "pred.png"
        mv_path = td_path / "pred_mv.png"
        gif_path = td_path / "pred.gif"
        save_structure_image(pred.coords, str(img_path), title=f"Predicted {pred.sequence}")
        save_multiview_image(pred.coords, str(mv_path), title=f"Predicted {pred.sequence}")
        save_rotation_gif(pred.coords, str(gif_path), title=f"Predicted {pred.sequence}", frames=24, fps=8)

        st.image(str(img_path), caption="Single-view render")
        st.image(str(mv_path), caption="Multi-view render")
        st.image(str(gif_path), caption="Animated rotation")

    ens = st.session_state.model.predict_ensemble(pred.sequence, n_members=int(ensemble_n))
    st.write("Ensemble summary")
    st.json({
        "members": ens["members"],
        "mean_confidence": float(ens["mean_confidence"].mean()),
        "coord_variance_mean": float(ens["coord_var"].mean()),
    })

    pdb_text = pred.to_pdb()
    st.download_button(
        "Download predicted PDB",
        data=pdb_text,
        file_name=f"{pred.sequence[:10]}_prediction.pdb",
        mime="chemical/x-pdb",
    )


def _exports_tab() -> None:
    st.subheader("Model Export")
    if st.button("Export model checkpoint (JSON)"):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.json"
            st.session_state.model.save(str(p))
            payload = p.read_text()
        st.download_button("Download checkpoint", payload, file_name="model.json", mime="application/json")

    st.markdown("---")
    st.write("Current model/config snapshot")
    st.code(json.dumps(st.session_state.model.config.__dict__, indent=2))


def main() -> None:
    _init_state()
    _render_header()
    _render_model_controls()

    t1, t2, t3 = st.tabs(["Train", "Predict + Visualize", "Export"])
    with t1:
        _train_tab()
    with t2:
        _infer_tab()
    with t3:
        _exports_tab()


if __name__ == "__main__":
    main()
