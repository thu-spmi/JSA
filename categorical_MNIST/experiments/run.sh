seed=$1

python jsa.py --seed $seed
python vimco.py --seed $seed
python gumbel.py --seed $seed
python arsm.py --seed $seed
