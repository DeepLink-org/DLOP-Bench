# Copyright (c) OpenComputeLab. All Rights Reserved.

# Copyright (c) OpenComputeLab. All Rights Reserved.

for file in `ls .`; do
  if [ -d $file ]; then
    cd $file
    for subfile in `ls $PWD`; do
      if [ -f $subfile ]; then
        sed -i '1 i # Copyright (c) OpenComputeLab. All Rights Reserved.\n' $subfile
      fi
      if [ -d $subfile ]; then
        cd $subfile
        for subsubfile in `ls $PWD`; do
            if [ -f $subsubfile -a ${subsubfile##*.} = "py" ]; then
               sed -i '1 i # Copyright (c) OpenComputeLab. All Rights Reserved.\n' $subsubfile
            fi
        done
        cd ..
      fi
    done
    cd ..
  fi
  if [ -f $file ]; then
    sed -i '1 i # Copyright (c) OpenComputeLab. All Rights Reserved.\n' $file
  fi
done



