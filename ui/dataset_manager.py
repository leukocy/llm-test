import pandas as pd
import streamlit as st

from core.dataset_loader import DatasetLoader


def render_dataset_manager():
    """Render the Dataset Management UI."""
    st.header("📂 Dataset Management")

    loader = DatasetLoader()

    # --- Upload Section ---
    with st.expander("📤 Upload New Dataset", expanded=False):
        uploaded_file = st.file_uploader("Select a CSV or JSON file", type=['csv', 'json'])
        if uploaded_file is not None and st.button("Save Dataset"):
            with st.spinner("Validating and saving..."):
                error = loader.save_dataset(uploaded_file, uploaded_file.name)
                if error:
                    st.error(error)
                else:
                    st.success(f"Dataset {uploaded_file.name} Saved successfully!")
                    st.rerun()

        st.info("💡 Dataset must contain a 'prompt' column. Optional columns: 'expected_output', 'id'.")

    st.markdown("---")

    # --- List & Manage Section ---
    st.subheader("Saved Datasets")

    datasets = loader.list_datasets()

    if not datasets:
        st.info("No datasets yet. Please upload a new dataset.")
    else:
        # Create a DataFrame for better display
        df_datasets = pd.DataFrame(datasets)

        # Showing as a table with actions
        for _index, row in df_datasets.iterrows():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.markdown(f"**📄 {row['filename']}**")
            with col2:
                st.caption(f"{row['type']} • {row['size']}")
            with col3:
                if st.button("👀 Preview", key=f"preview_{row['filename']}"):
                    st.session_state.preview_dataset = row['filename']
            with col4:
                if st.button("🗑️ Delete", key=f"delete_{row['filename']}", type="secondary"):
                    if loader.delete_dataset(row['filename']):
                        st.success(f"Deleted {row['filename']}")
                        if 'preview_dataset' in st.session_state and st.session_state.preview_dataset == row['filename']:
                            del st.session_state.preview_dataset
                        st.rerun()
                    else:
                        st.error("Delete failed")

    # --- Preview Section ---
    if 'preview_dataset' in st.session_state:
        st.markdown("---")
        st.subheader(f"👀 Preview: {st.session_state.preview_dataset}")
        preview_df = loader.get_dataset_preview(st.session_state.preview_dataset)
        if preview_df is not None:
            st.dataframe(preview_df, width="stretch")
            st.caption(f"Showing first {len(preview_df)} rows")
        else:
            st.error("Unable to load preview")
