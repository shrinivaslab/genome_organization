import os
import subprocess

file_name = "/Users/kadendimarco/Desktop/Shrivinas_lab/genome_archetecture/Megaenhancers/experimental_data/4DNFISZ88WZA.hic"
folder_name = "intra_top100_ME"
loops_file = "/Users/kadendimarco/Desktop/Shrivinas_lab/genome_archetecture/Megaenhancers/experimental_data/ME_loop.txt"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

## 25Kb, 1Mb window
#command = f"java -jar /Users/kadendimarco/Desktop/Shrivinas_lab/genome_archetecture/utilis/tools/juicer/scripts/common/juicer_tools.jar apa -k KR -r 25000 -n 30 -w 20 {file_name} {loops_file} {folder_name}"
command = f"java -jar /Users/kadendimarco/Desktop/Shrivinas_lab/genome_archetecture/utilis/tools/juicer/scripts/common/juicer_tools.jar apa -k KR -r 25000 -n 0 -w 20 {file_name} {loops_file} {folder_name}"
subprocess.run(command, shell=True, check=True)