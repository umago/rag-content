ARG EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

FROM registry.access.redhat.com/ubi9/python-311 as lightspeed-rag-builder
ARG EMBEDDING_MODEL

USER 0
WORKDIR /workdir

COPY pyproject.toml pdm.lock Makefile .
RUN make install-tools && pdm config python.use_venv false && make pdm-lock-check install-deps

COPY ocp-product-docs-plaintext ./ocp-product-docs-plaintext
COPY runbooks ./runbooks

COPY scripts/download_embeddings_model.py .
RUN pdm run python download_embeddings_model.py -l ./embeddings_model -r ${EMBEDDING_MODEL}

COPY scripts/generate_embeddings.py .
RUN set -e && for OCP_VERSION in $(ls -1 ocp-product-docs-plaintext); do \
        pdm run python generate_embeddings.py -f ocp-product-docs-plaintext/${OCP_VERSION} -r runbooks/alerts -md embeddings_model \
            -mn ${EMBEDDING_MODEL} -o vector_db/ocp_product_docs/${OCP_VERSION} \
            -i ocp-product-docs-$(echo $OCP_VERSION | sed 's/\./_/g') -v ${OCP_VERSION}; \
    done

FROM registry.access.redhat.com/ubi9/ubi-minimal@sha256:104cf11d890aeb7dd5728b7d7732e175a0e4018f1bb00d2faebcc8f6bf29bd52
COPY --from=lightspeed-rag-builder /workdir/vector_db/ocp_product_docs /rag/vector_db/ocp_product_docs
COPY --from=lightspeed-rag-builder /workdir/embeddings_model /rag/embeddings_model

# this directory is checked by ecosystem-cert-preflight-checks task in Konflux
RUN mkdir /licenses
COPY LICENSE /licenses/

# Labels for enterprise contract
LABEL com.redhat.component=openshift-lightspeed-rag-content
LABEL description="Red Hat OpenShift Lightspeed RAG content"
LABEL distribution-scope=private
LABEL io.k8s.description="Red Hat OpenShift Lightspeed RAG content"
LABEL io.k8s.display-name="Openshift Lightspeed RAG content"
LABEL io.openshift.tags="openshift,lightspeed,ai,assistant,rag"
LABEL name=openshift-lightspeed-rag-content
LABEL release=0.0.1
LABEL url="https://github.com/openshift/lightspeed-rag-content"
LABEL vendor="Red Hat"
LABEL version=0.0.1
LABEL summary="Red Hat OpenShift Lightspeed RAG content"

USER 65532:65532