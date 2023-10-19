import MDAnalysis as mda
from Bio import SeqIO


def get_label_seq(fasta_file):
    record = SeqIO.read(fasta_file, "fasta")
    return record.name, record.seq


def batcher(iterable, n=1):
    len_iter = len(iterable)
    for ndx in range(0, len_iter, n):
        yield iterable[ndx : min(ndx + n, len_iter)]


def get_plddt(pdb_file):
    mda_u = mda.Universe(pdb_file)
    return mda_u.atoms.bfactors.mean()
