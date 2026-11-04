"""ESM-2 protein embeddings for the perturbed genes: the table model.perturbation_knowledge=esm reads.

    python scripts/build_esm_knowledge.py --model <folder holding facebook/esm2_t33_650M_UR50D>

INPUT. assets/knowledge/norman_uniprot.fasta is UniProt's reviewed human entry for each of
Norman's 101 perturbed genes, one per gene, matched on the header's GN= field. Query:
gene_exact:<gene> AND organism_id:9606 AND reviewed:true. It is committed, so the table
can be rebuilt without the network.

OUTPUT. assets/knowledge/norman_esm2_650m.npz. It holds, per gene:
- genes: the gene symbol
- vectors: [n, 1280], the last layer's per-residue states averaged over the residues
  (the BOS and EOS tokens left out)
- accessions and lengths: provenance

A protein longer than the model's 1,022-residue window is cut into consecutive windows.
Each window is encoded on its own and the residue states are averaged over all windows,
so every residue counts once. The reduction to the components the encoder uses happens
when the table is read (src/data/knowledge.py, esm_embedding), not here.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import torch

WINDOW = 1022  # residues per pass: ESM-2's 1,024 positions minus BOS and EOS


def read_fasta(path: str) -> dict[str, tuple[str, str]]:
    """gene -> (accession, sequence), from UniProt headers 'sp|ACC|NAME ... GN=GENE ...'."""
    entries: dict[str, tuple[str, str]] = {}
    gene = accession = None
    parts: list[str] = []

    def flush() -> None:
        if gene is not None:
            if gene in entries:
                raise ValueError(f"{path}: {gene} appears twice")
            entries[gene] = (accession, "".join(parts))

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                found = re.search(r" GN=(\S+)", line)
                if found is None:
                    raise ValueError(f"{path}: no GN= in {line[:60]}")
                gene, accession, parts = found.group(1), line.split("|")[1], []
            else:
                parts.append(line)
    flush()
    return entries


@torch.no_grad()
def embed(sequence: str, tokenizer, model, device: str) -> np.ndarray:
    """The mean over residues of the last layer, windows of WINDOW residues."""
    total = None
    for start in range(0, len(sequence), WINDOW):
        window = sequence[start:start + WINDOW]
        tokens = tokenizer(window, return_tensors="pt").to(device)
        states = model(**tokens).last_hidden_state[0, 1:-1]       # drop BOS and EOS
        if states.shape[0] != len(window):
            raise ValueError(f"{states.shape[0]} states for {len(window)} residues")
        block = states.double().sum(dim=0).cpu().numpy()
        total = block if total is None else total + block
    return (total / len(sequence)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="local folder with facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--name", default="facebook/esm2_t33_650M_UR50D",
                        help="the checkpoint's hub name, stored in the table")
    parser.add_argument("--fasta", default="assets/knowledge/norman_uniprot.fasta")
    parser.add_argument("--out", default="assets/knowledge/norman_esm2_650m.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from transformers import AutoTokenizer, EsmModel

    entries = read_fasta(args.fasta)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmModel.from_pretrained(args.model, use_safetensors=True,
                                     add_pooling_layer=False).to(args.device).eval()
    genes = sorted(entries)
    vectors = []
    for i, gene in enumerate(genes):
        accession, sequence = entries[gene]
        vectors.append(embed(sequence, tokenizer, model, args.device))
        print(f"  {i + 1:3d}/{len(genes)}  {gene:10s} {accession}  {len(sequence):5d} aa")
    vectors = np.stack(vectors)
    if not np.isfinite(vectors).all():
        raise ValueError("a non-finite embedding")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, genes=np.array(genes), vectors=vectors,
        accessions=np.array([entries[g][0] for g in genes]),
        lengths=np.array([len(entries[g][1]) for g in genes]),
        model=np.array(args.name))
    print(f"-> {args.out}  {vectors.shape[0]} genes x {vectors.shape[1]}")


if __name__ == "__main__":
    main()
