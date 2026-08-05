#!/bin/sh
# Fetch the structures and alignments the calibration sweep runs on.
#
# ~40MB, not committed. Every accession has both an AlphaFold DB model and the
# a3m that model was built from, so the alignment and the ground truth agree by
# construction. They are E. coli proteins spanning 70-270 residues, all deeper
# than 8k sequences so that depth does not confound the length axis -- which is
# why P0ABE7 (128 residues but only 647 sequences) is deliberately absent.
set -e
cd "$(dirname "$0")"
mkdir -p data

LIST="P0A9X9 P0A6A8 P0A6F9 P0AG59 P0A7R5 P0AA25 P0AE67 P0A7Y4 P0A7B8 P0A805 P0A9K9 P0AEZ3"

for ACC in $LIST; do
  if [ -f "data/${ACC}.a3m" ] && [ -f "data/${ACC}.pdb" ]; then
    echo "$ACC  cached"
    continue
  fi
  mc=$(curl -sS -o "data/${ACC}.a3m" -w "%{http_code}" \
       "https://alphafold.ebi.ac.uk/files/msa/AF-${ACC}-F1-msa_v6.a3m")
  pc=$(curl -sS -o "data/${ACC}.pdb" -w "%{http_code}" \
       "https://alphafold.ebi.ac.uk/files/AF-${ACC}-F1-model_v6.pdb")
  if [ "$mc" != "200" ] || [ "$pc" != "200" ]; then
    rm -f "data/${ACC}.a3m" "data/${ACC}.pdb"
    echo "$ACC  FAILED (msa=$mc pdb=$pc)"
    continue
  fi
  nres=$(awk '/^ATOM/ && substr($0,13,4)==" CA "{n++} END{print n+0}' "data/${ACC}.pdb")
  nseq=$(grep -c '^>' "data/${ACC}.a3m")
  echo "$ACC  L=$nres  N=$nseq"
done
