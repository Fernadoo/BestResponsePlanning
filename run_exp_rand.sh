### small
# python experiments_rand.py --map small --cfg small --agents 2 \
#   	--tests 0 1 2 4 5 6 7 8 9 10 11 \
#   	--prefix exp/small_rand --num-sim 5e2 --oppo-type rational --plot --mp 8

# python experiments_rand.py --map small --cfg small --agents 2 \
#   	--tests 4 5 6 \
#   	--prefix exp/small_rand --num-sim 5e2 --oppo-type malicious --plot --mp 8

# python experiments_rand.py --map small --cfg small --agents 2 \
#   	--tests 0 1 2 4 5 6 7 8 9 10 11 \
#   	--prefix exp/small_rand --num-sim 5e2 --oppo-type self --plot --mp 8

### square
# python experiments_rand.py --map square --cfg square --agents 2 \
#   	--tests 4 5 6 \
#   	--prefix exp/square_rand --num-sim 1e3 --oppo-type rational --plot --mp 8

# python experiments_rand.py --map square --cfg square --agents 2 \
#   	--tests 4 5 6 \
#   	--prefix exp/square_rand --num-sim 1e3 --oppo-type malicious --plot --mp 8

# python experiments_rand.py --map square --cfg square --agents 2 \
#   	--tests 0 1 4 5 6 7 8 12 \
#   	--prefix exp/square_rand --num-sim 1e3 --oppo-type self --mp 8

### square4a
# python experiments_rand.py --map square --cfg square4a --agents 4 \
#   	--tests 7 8 \
#   	--prefix exp/square4a_rand --num-sim 15e2 --oppo-type rational --mp 8

# python experiments_rand.py --map square --cfg square4a --agents 4 \
#   	--tests 0 1 7 8 12 \
#   	--prefix exp/square4a_rand --num-sim 15e2 --oppo-type malicious --mp 8

# python experiments_rand.py --map square --cfg square4a --agents 4 \
#   	--tests 0 1 7 8 12 \
#   	--prefix exp/square4a_rand --num-sim 15e2 --oppo-type self --mp 8

### random60a
python experiments_rand.py --map random32 --cfg random50a --agents 50 \
  	--tests 0 1 12 \
  	--prefix exp/random60a_rand --num-sim 15e2 --oppo-type rational --mp 8

python experiments_rand.py --map random32 --cfg random50a --agents 50 \
  	--tests 0 1 12 \
  	--prefix exp/random60a_rand --num-sim 15e2 --oppo-type malicious --mp 8

python experiments_rand.py --map random32 --cfg random50a --agents 50 \
  	--tests 0 1 12 \
  	--prefix exp/random60a_rand --num-sim 15e2 --oppo-type self --mp 8

### medium20a
# python experiments_rand.py --map medium --cfg medium20a --agents 20 \
#   	--tests 7 8 \
#   	--prefix exp/medium20a_rand --num-sim 2e3 --oppo-type rational --mp 8

# python experiments_rand.py --map medium --cfg medium20a --agents 20 \
#   	--tests 0 1 7 8 12 \
#   	--prefix exp/medium20a_rand --num-sim 2e3 --oppo-type malicious --mp 8

# python experiments_rand.py --map medium --cfg medium20a --agents 20 \
#   	--tests 12 \
#   	--prefix exp/medium20a_rand --num-sim 2e3 --oppo-type self --mp 8
