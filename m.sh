inputs=(
    "roberta-large-nsp-1000000-1e-06-32"
    "roberta-large-pp-500000-1e-06-128"
    "roberta-large-dual-500000-1e-06-128"
    "roberta-base-pp-1000000-1e-06-128"
    "roberta-base-nsp-1000000-1e-06-32"
    "roberta-base-dual-1000000-1e-06-128"
)

for input in "${inputs[@]}"; do 
    python3 con.py --path "$input" 
done