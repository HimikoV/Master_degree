import subprocess


num_pop = []
num_gen = []
num_exp = 1000
file_bat = 'start.bat'
file_sh = 'start.sh'


def bat_overrite(bat_file, x, y, z):
    with open(bat_file, "w") as dft:
        dft.truncate()
        # cd to location of your algorithm
        dft.write(
            f"python main.py --train --no-show --num-pop {x} --num-gen {y} --num-exp {z}")
    return


for i in range(10):
    num_pop.append(10*(i+1))

for i in range(100):
    num_gen.append(10*(i+1))


for i in num_pop:
    for k in num_gen:
        bat_overrite(file_sh, x=i, y=k, z=num_exp)
        subprocess.call([file_sh])
