# create training data

python naf.py Pendulum-v0 Pendulum_Simple_3 --episodes=1000 --no_display
python naf.py PendulumLong-v0 Pendulum_Long_3 --episodes=1000 --no_display
python naf.py PendulumHeavy-v0 Pendulum_Heavy_3 --episodes=1000 --no_display
# python naf.py PendulumLongHeavy-v0 Pendulum_Long_Heavy_3 --episodes=200 --no_display