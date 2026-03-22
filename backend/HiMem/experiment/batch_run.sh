ENV=test
#ENV=test_without_ka
SEARCH_MODE=hybrid
#NAME=before_knowledge_conflict_resolution
NAME=None

top_ks=(5 10 15 20 25)
rounds=(1 2 3)

for top_k in ${top_ks[@]}
do
    for round in ${rounds[@]}
    do
        python experiment/gen.py --env ${ENV} --search_mode ${SEARCH_MODE} --top_k ${top_k} --round ${round} --name=${NAME}
        python experiment/benchmark.py --env ${ENV} --search_mode ${SEARCH_MODE} --top_k ${top_k} --round ${round} --name=${NAME}
    done
done