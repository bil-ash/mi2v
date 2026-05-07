#/bin/bash
set -e

work_dir=output_flow_server/debug
np=8


if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    config="./configs/mobilei2v_config/MobileI2V_300M_img512.yaml"
    echo "Only support .yaml files, but get $1. Set to --config_path=$config"
fi

TRITON_PRINT_AUTOTUNING=1 \
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=$np --master_port=15849 \
        train_scripts/train_i2v.py \
        --config_path=$config \
        --work_dir=$work_dir \
        --name=tmp \
        --report_to=tensorboard \
        --debug=true \
        "$@"