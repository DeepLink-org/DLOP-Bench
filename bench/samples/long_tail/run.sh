for file in `ls .`; do
  if [ -d $file ]; then
    cd $file
    if [ -f pat_impl.py -a -f torch_impl.py ]; then
      rm pat_impl.py 
    elif [ -f pat_impl.py -a ! -f torch_impl.py ]; then
      mv pat_impl.py torch_impl.py
    fi
    for subfile in `ls $PWD`; do
      # echo $PWD
      # echo "=="
      if [ -f $subfile -a $(basename $subfile) = "torch_impl.py" ]; then
        echo "done"
      elif [ -f $subfile -a $(basename $subfile) = "xla_impl.py" ]; then
        echo "done"
      elif [ -f $subfile ]; then
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
     if [ $(basename $file) = "__init__.py" ]; then
        sed -i '1 i # Copyright (c) OpenComputeLab. All Rights Reserved.\n' $file
     fi
  fi
done



