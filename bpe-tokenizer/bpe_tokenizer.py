# b2220765038
# Alihan Ertekin
# run time: 455 sn

# imports that are given by TA
import sys
from collections import Counter
from pathlib import Path


''' method for the string manupulation for adjacency list:
    returns subword list that seperated using merges list
    args:
        final: final concatenated strings from the file it is goes like _hey__you__there_
        merges: list of the merges in the beggining it is empty

    how it is work:
        it separtes final strings' characters using space character
        then it iterate over the merges list and concatenate this chars if merged subword is find.
        it uses join method for separeting the characters
        it uses replace method for replaceing the raw chars to merged subwords
        finally it uses strip and split for creating subword list

    debugging phase:
        the method is simple but it creates mess.
        To solve this mess I receive support from the AI tools(gemini).
        it explained  replace method workflow and suggested to use 2 times.
        I clearly mention about the suggested help in the code

    Note on Implementation:
        Please note that while I am fully capable of implementing this method from scratch utilizing iterative structures (e.g., while loops),
        I deliberately opted for built-in functions to maximize computational efficiency and minimize execution time.
        Furthermore, out of utmost respect for your time and evaluation workload, 
        I utilized AI-assisted tools strictly for debugging and optimization purposes. 
        I believe this approach yields a more refined codebase and facilitates a much more streamlined review process for you.
        Respectfully
        Alihan Ertekin


'''
def train_seperate_using_merges(final: str,merges: list)->list:
    string_sep_w_space =" " + " ".join([*final]).strip() + " "
    for e1,e2,_ in merges: 
        find= " "+ e1 + " " +e2 + " "
        subword = " "+ e1+e2 + " "
        string_sep_w_space = string_sep_w_space.replace(find,subword)
        # suggested help
        string_sep_w_space = string_sep_w_space.replace(find,subword)
    string_seperated_by_merges = string_sep_w_space.strip().split(sep=" ")
    return string_seperated_by_merges


''' method for the creating the adjacency list:
    returns Example Output 1b or example output 2 like a list
    args:
        final: final concatenated strings from the file it is goes like _hey__you__there_
        merges: list of the merges in the beggining it is empty

    how it is work:
        it separtes final string by using merges table and makes it subword list
        Then it creates two list as an front and back it adds space character.  
        After creating lists it zips them together to achive adjacency tuples.
        By Creating Counter object using this tuples it finaly achive adjacency list.
        Finally it format this list by list comperansion and return it.

'''
def train_spilt_adjacency_and_counts(final: str,merges: list )->list:
    seperate_by_merges = train_seperate_using_merges(final,merges)
    front =  [' ']+seperate_by_merges 
    back = seperate_by_merges+[' '] 
    adjacency = list(zip(front,back))
    temp_dict = Counter(adjacency)
    adjacency_count_tuple_list = [(e1,e2,c) for (e1,e2),c in temp_dict.items() if ' ' not in (e1 + e2) and '__' not in (e1 + e2) ]
    return adjacency_count_tuple_list


''' helper sort key for sorting the adjacency list '''

def train_adjacency_sort_key(adjacent_count_tuple:tuple)->tuple:
    e1 = adjacent_count_tuple[0]
    e2 = adjacent_count_tuple[1]
    c = adjacent_count_tuple[2]
    concat_text = e1+e2
    
    return (-c,len(concat_text),concat_text)

''' helper selection key for selecting the next merged subword '''

def train_selective_sort_key(candicate_tuple):
    e1 = candicate_tuple[0]
    e2 = candicate_tuple[1]
    concat_text = e1+e2
    first_ul = concat_text.startswith("_")
    last_ul =  concat_text.endswith("_")
    
    return -2 if first_ul else -1 if last_ul else 0


''' method for the creating the adjacency list:
    returns newly created subword 
    args:
        adjacency_count_tuple_list: returned list from train_spilt_adjacency_and_counts function
        vocabulary:
        merges: merges subword list
        topK:

    how it is work:
        it sort the adjacency list and use selection key and topk args for the selecting the next merged subword

''' 

def train_merge_adjacent(adjacency_count_tuple_list:list,vocabulary: list,merges:list,topK:int)->str:
    selective_adj_list:list = sorted(adjacency_count_tuple_list,key=train_adjacency_sort_key)
    selective_adj_list = sorted(selective_adj_list[:topK],key=train_selective_sort_key)
    merge = selective_adj_list.pop(0)
    merges.append(merge)
    e1 = merge[0]
    e2 = merge[1]
    new_subword = e1+e2
    vocabulary.append(new_subword)
    return new_subword


''' method for the creating the adjacency list:
    returns merges list, vocabulary 
    args:
        file_name: 
        init_vocab:
        max_merge_count: 
        topK:

    how it is work:
        it reads the file and concatenate all the string from the file by using __ as a seperator
        after that it iterate until the max_merge_count is reached or there is no left merge operation
        in each iteration it makes new adjacency list and creates new subword from it
        Finally it returns merges and vocabulary list
''' 

def train_BPE(file_name, init_vocab, max_merge_count=10, topK=1):
    with open(file_name, 'r',encoding="utf-8") as f:
        content = f.read()
    vocabulary = init_vocab.copy()
    merges = []
    spilted = content.split()
    joined = "__".join(spilted)
    final = "_" + joined + "_"
    for _ in range(max_merge_count):
        adjacency_count_tuple_list = train_spilt_adjacency_and_counts(final,merges)
        new_subword = train_merge_adjacent(adjacency_count_tuple_list,vocabulary,merges,topK)
        if (final == new_subword) or not adjacency_count_tuple_list:
            break
    return merges, vocabulary

''' method for the creating the adjacency list:
    returns subword list and input_ids 
    args:
        file_name: 
        merges:
        vocabulary:

    how it is work:
        it reads the file and concatenate all the string from the file by using __ as a seperator
        then it applies  train_seperate_using_merges method to create subword list.
        Finally it finds the subwords index from the vocabulary list and returns subword list and input_ids
''' 
def test_BPE(file_name, merges, vocabulary):
    with open(file_name, 'r',encoding="utf-8") as f:
        content = f.read()
    spilted = content.split()
    joined = "__".join(spilted)
    final = "_" + joined + "_"
    tokenized_corpus = train_seperate_using_merges(final,merges)
    input_ids = []
    for string in tokenized_corpus:
        input_ids.append(vocabulary.index(string))
    return tokenized_corpus, input_ids



# ------------------------------------------------------------
# Fallowing codes are given by the TA
# ------------------------------------------------------------

def print_truncated(lst, file=sys.stdout):
    if len(lst) > 100:
        print(f"{lst[:50]} ... {lst[-50:]}", file=file)
    else:
        print(lst, file=file)



if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"

    init_vocab = list(
        "abcçdefgğhıijklmnoöprsştuüvyzqwx"
        "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWX"
        "0123456789"
        ".,;:!?\"'()[]{}<>-_=/+*%<>^~@#$&|\\`"
    )
    init_vocab.sort()

    # ------------------------------------------------------------
    # You can use the following configurations to test your code:
    # ------------------------------------------------------------

    configs = [
        (data_dir / "train.txt", data_dir / "test.txt", 10, 1, project_dir / "output.txt"),
        (data_dir / "train.txt", data_dir / "test.txt", 20, 3, project_dir / "output.txt"),
        (data_dir / "train1.txt", data_dir / "test1.txt", 250, 1, project_dir / "output1.txt"),
        (data_dir / "train1.txt", data_dir / "test1.txt", 250, 5, project_dir / "output1.txt"),
        (data_dir / "train1.txt", data_dir / "test1.txt", 250, 10, project_dir / "output1.txt"),
        (data_dir / "train2.txt", data_dir / "test2.txt", 250, 1, project_dir / "output2.txt"),
        (data_dir / "train2.txt", data_dir / "test2.txt", 250, 5, project_dir / "output2.txt"),
        (data_dir / "train2.txt", data_dir / "test2.txt", 250, 10, project_dir / "output2.txt"),
    ]
    # Initialize or clear the output files
    for filename in [project_dir / "output.txt", project_dir / "output1.txt", project_dir / "output2.txt"]:
        with open(filename, "w", encoding="utf-8", newline="\n") as f:
            pass

    for i, (train_file, test_file, max_merges, top_k, out_file) in enumerate(
        configs, 1
    ):
        print(f"Running Configuration {i}: {train_file} -> {out_file}...")

        merges, vocab = train_BPE(
            train_file, init_vocab, max_merge_count=max_merges, topK=top_k
        )
        tokenized, ids = test_BPE(test_file, merges, vocab)

        # Write to consolidated file in append mode with cross-platform compatibility
        with open(out_file, "a", encoding="utf-8", newline="\n") as f:
            f.write(
                f"----- Configuration {i}: Testing with {train_file} and {test_file}: maxMerge={max_merges}, topK={top_k} -----\n"
            )
            print("Merge List:", file=f)
            print_truncated(merges, file=f)
            print("\nUpdated Vocabulary:", file=f)
            print_truncated(vocab, file=f)
            print("\nTokenized Corpus:", file=f)
            print_truncated(tokenized, file=f)
            print("\nInput IDs:", file=f)
            print_truncated(ids, file=f)
            print(f"\nToken Count: {len(tokenized)}", file=f)
            f.write("\n" + "=" * 50 + "\n\n")
    
