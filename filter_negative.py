# -*- coding: utf-8 -*-
from Bio import SeqIO


def filter():
    file_name = 'data/filtered_negative.fasta'
    rec_non = list(SeqIO.parse("data/negative.fasta", "fasta"))
    with open(file_name, 'a', encoding="utf-8") as myfile:
        for i, item in enumerate(rec_non):
            if check_non_coding(item.description):
                if len(item.seq) > 1500:
                    list_chunks = split_sequence(str(item.seq))
                    for j, chunk in enumerate(list_chunks):
                        myfile.write(f">{item.description}-{j}\n")
                        myfile.write(f"{chunk}\n")
                else:
                    myfile.write(f">{item.description}\n")
                    myfile.write(f"{item.seq}\n")


def split_sequence(sequence):
    chunk_size = 1500
    return [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]


def check_non_coding(item_description):
    list_tags = ["RNase H", "spartic", "Gag", "ntegras", "ranscriptas", "ranspos", "olyprotein",
                 "virus", "chromodomain", "nvelop", "copia", "gypsy", "protein"]
    for tag in list_tags:
        if tag in item_description:
            return False
    return True


def analyze_filtered_negative():
    rec_non_filtered = list(SeqIO.parse("data/filtered_negative.fasta", "fasta"))
    size_list = [len(x.seq) for x in rec_non_filtered]
    print(f"The total amount of negative sequences after filtration is {len(rec_non_filtered)}")
    print(f"The minimum size of the filtered negative sequences is {min(size_list)}")
    print(f"The maximum size of the filtered negative sequences is {max(size_list)}")


def main():
    filter()
    analyze_filtered_negative()


if __name__ == "__main__":
    main()
