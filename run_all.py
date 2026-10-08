from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for args in [['-m','unittest','discover','-s',str(root),'-p','test_model.py'],[str(root/'run_experiments.py')],[str(root/'threshold_study.py')]]:
 subprocess.run([sys.executable,*args],cwd=root,check=True)
print('All tests and experiments completed. Open results/summary.json and results/*.png.')
