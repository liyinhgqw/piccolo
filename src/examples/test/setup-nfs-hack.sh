#!/bin/bash

set -x

mkdir -p /scratch/cp-union
fusermount -u /scratch/cp-union

FUSE_CMD=

for h in 2 3 5 6 7 8 9 10 11 12 13; do
  FUSE_CMD="/scratch/cp-remote/beaker-$h=RW:${FUSE_CMD}"

  if [[ ! -d /scratch/cp-remote/beaker-$h ]]; then
    mkdir -p /scratch/cp-remote/beaker-$h
  fi

  umount -l /scratch/cp-remote/beaker-$h
  if [[ $1 -ne "down" ]]; then
    mount beaker-$h:/scratch/ /scratch/cp-remote/beaker-$h
  fi
done

if [[ $1 -ne "down" ]]; then
  unionfs-fuse  -o allow_other -o uid=1043 -o gid=1043 $FUSE_CMD /scratch/cp-union
fi
