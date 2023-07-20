# Debian-based distributions
arch=$(dpkg --print-architecture)
curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v3.4.1/enroot-hardened_3.4.1-1_${arch}.deb
curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v3.4.1/enroot-hardened+caps_3.4.1-1_${arch}.deb # optional
sudo apt install -y ./*.deb

