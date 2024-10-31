#!/usr/bin/zsh

echo "Updating the datago binaries"

# Get the current python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Building package for python" $python_version

# Setup where the python package will be copied
DESTINATION="../../../python_$python_version"
rm -rf $DESTINATION

# Build the python package via the gopy toolchain
cd pkg
gopy pkg -author="Photoroom" -email="team@photoroom.com" -url="" -name="datago" -version="0.3" .
mkdir -p $DESTINATION/datago
mv datago/* $DESTINATION/datago/.
mv setup.py $DESTINATION/.
mv Makefile $DESTINATION/.
mv README.md $DESTINATION/.
rm LICENSE
rm MANIFEST.in

cd ..
