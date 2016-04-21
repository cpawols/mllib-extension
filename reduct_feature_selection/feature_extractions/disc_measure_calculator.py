from collections import Counter


# TODO: add tests
class DiscMeasureCalculator(object):

    @staticmethod
    def prepare_hist(decisions):
        dec_fqs = Counter()
        for dec in decisions:
            dec_fqs[dec] += 1
        return {k: (0, v) for k, v in dec_fqs.iteritems()}

    @staticmethod
    def update_award(dec, act_left_sum, act_right_sum, dec_fqs_disc, act_award):
        act_award += (act_right_sum - act_left_sum - 1) - \
                     (dec_fqs_disc[dec][1] - dec_fqs_disc[dec][0] - 1)
        act_left_sum += 1
        act_right_sum -= 1
        dec_fqs_disc[dec] = (dec_fqs_disc[dec][0] + 1,
                             dec_fqs_disc[dec][1] - 1)
        return act_left_sum, act_right_sum, act_award
