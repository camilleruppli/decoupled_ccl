import json
from collections import Counter


def group_pirads(score):
    if score < 3:
        score = 0
    elif score > 3:
        score = 1
    return score


def new_annotation_db_with_confidence(db_metadata, filename):
    # db metadata is a json in which metadata scores are stored after radiologists
    # annotations
    # for example db_metadata = {"id":[2,2,2,5]}
    annot_num = 0
    count = 0
    db_annot_confidence = {}
    for mongo_id in db_metadata.db:
        if len(db_metadata.db[mongo_id]) != 1:
            annot_num += len(db_metadata.db[mongo_id])
            count += 1
    mean_annot = annot_num / count
    for mid in db_metadata:
        pirads_list = [group_pirads(annot) for annot in db_metadata[mid]
                       if annot != 3]
        if len(pirads_list) == 1:
            if pirads_list[0] == 1:
                annot = [-1, -1]
            else:
                annot = [pirads_list[0], 1 / mean_annot]
        elif len(pirads_list) == 0:
            annot = [-1, -1]
        else:
            annotator_number = len(pirads_list)
            majority_voting = Counter(pirads_list).most_common(1)[0][0]
            confidence = sum([p == majority_voting for p in pirads_list]) / annotator_number
            assert 0.5 <= confidence <= 1, print(majority_voting, pirads_list, annotator_number)
            annot = [majority_voting, confidence]
        db_annot_confidence[mid] = annot
    with open(filename, "r") as f:
        json.dump(db_annot_confidence, f)
