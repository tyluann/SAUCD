import json

max_round = 6


def finish_check_db(db, Annot, ctrl):
    total_score = 0; field_count = 0
    for i in range(len(ctrl.paras['ids'])):
        id = ctrl.paras['ids'][i]
        annot = db.session.query(Annot).get(id)
        for field in sorted(annot.__dict__):
            if field == 'id' or field == '_sa_instance_state':
                continue
            score = getattr(annot, field)
            if score is not None:
                total_score += score
                field_count += 1
        if not field_count * int(max_round / 2) == total_score:
            return False
    return True


def error_check_db(db, Annot, ctrl):
    sqr_e = 0; count = 0
    avg_score_file = 'userstudy/static/avg_score.json'
    with open(avg_score_file, 'r') as f:
        avg_score = json.load(f)
    for i in range(len(ctrl.paras['ids'])):
        id = ctrl.paras['ids'][i]
        obj = id.split('_')[0]
        annot = db.session.query(Annot).get(id)
        for field in sorted(annot.__dict__):
            if field == 'id' or field == '_sa_instance_state':
                continue
            score = getattr(annot, field)
            if score is not None:
                sqr_e += (score - avg_score[obj]['avg_score'][field]) ** 2
                count += 1
        sqr_e /= count
    return sqr_e