ENV=test
#ENV=test_without_ka
CONSTRUCTION_MODE=all
# search-mode can be chosen from: hybrid|best-effort|note|episode
SEARCH_MODE=hybrid
TOP_K=10
ROUND=1
#NAME=before_knowledge_conflict_resolution
#NAME=enable_knowledge_alignment
NAME=None
# Construct memory
#python experiment/memory_construction.py --env ${ENV} --construction_mode ${CONSTRUCTION_MODE}
#python experiment/memory_construction.py --env ${ENV} --construction_mode ${CONSTRUCTION_MODE} --enable_knowledge_alignment

# Generate the answer based on the search results
python experiment/gen.py --env ${ENV} --search_mode ${SEARCH_MODE} --top_k ${TOP_K} --round ${ROUND} --name=${NAME}

# Evaluation
python experiment/benchmark.py --env ${ENV} --search_mode ${SEARCH_MODE} --top_k ${TOP_K} --round ${ROUND} --name=${NAME}