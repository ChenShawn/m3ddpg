python train.py --scenario simple_tag --max-episode-len 50 --num-adversaries 2

# benchmark m3ddpg vs m3ddpg
python train.py --scenario simple_tag --max-episode-len 50 --num-adversaries 2 --good-policy mmmaddpg --bad-policy mmmaddpg --load-good ./m3ddpg_vs_m3ddpg/ --load-bad ./m3ddpg_vs_m3ddpg/ --display

# benchmark m3ddpg (good) vs ddpg (bad)
python train.py --scenario simple_tag --max-episode-len 50 --num-adversaries 2 --good-policy mmmaddpg --bad-policy ddpg --load-good ./m3ddpg_vs_m3ddpg/ --load-bad ./ddpg_vs_ddpg/ --display
