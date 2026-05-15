"""Training subsystem.

Contains all components that build the adaptive knowledge base from
historical XML estimates:

* ``xml_parser``        – parse SMR-Pro XML and form PTM transactions
* ``train_fp_growth``   – build association rules
* ``statistics_builder``– compute statistical profiles
* ``train_sbert``       – fine-tune SBERT
* ``train_fasttext``    – train FastText on rascenka names
* ``build_indices``     – build BM25 + FAISS indices + SymSpell dict
* ``pipeline``          – run all of the above in order
"""
