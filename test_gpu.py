import subprocess; process = subprocess.Popen('nvidia-smi -i 0 --query-gpu=name --format=csv'.split(), stdout=subprocess.PIPE); output, _ = process.communicate(); output = str(output); device_capability_map = {
    'Tesla K80'   : '37',
    'Tesla K40'   : '35',
    'Tesla K20'   : '35',
    'Tesla C2075' : '20',
    'Tesla C2050' : '20',
    'Tesla C2070' : '20',
    'Tesla V100'  : '70',
    'Tesla P100'  : '60',
    'Tesla P40'   : '61',
    'Tesla P4'    : '61',
    'Tesla M60'   : '52',
    'Tesla M40'   : '52',
    'Tesla K80'   : '37',
    'Tesla K40'   : '35',
    'Tesla K20'   : '35',
    'Tesla K10'   : '30',
    'GeForce GTX 1080 Ti' : '61'
}; cap = '61';
for k, v in device_capability_map.items():
    if k in output:
        cap = v
        break
print(output)
print(cap)
