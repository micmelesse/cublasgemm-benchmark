apt update
apt -y install git
rm -rf nvidia_sdk_samples
git clone https://github.com/pathscale/nvidia_sdk_samples
cd nvidia_sdk_samples