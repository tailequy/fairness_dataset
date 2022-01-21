# encoding: utf-8

import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy
import matplotlib.pyplot as plt


def calculate_performance(data, labels, predictions, probs, saIndex, saValue):
    protected_pos = 0.
    protected_neg = 0.
    non_protected_pos = 0.
    non_protected_neg = 0.

    tp_protected = 0.
    tn_protected = 0.
    fp_protected = 0.
    fn_protected = 0.

    tp_non_protected = 0.
    tn_non_protected = 0.
    fp_non_protected = 0.
    fn_non_protected = 0.
    for idx, val in enumerate(data):
        # protrcted population
        if val[saIndex] == saValue:
            if predictions[idx] == 1:
                protected_pos += 1.
            else:
                protected_neg += 1.


            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_protected += 1.
                else:
                    tn_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_protected += 1.
                else:
                    fp_protected += 1.

        else:
            if predictions[idx] == 1:
                non_protected_pos += 1.
            else:
                non_protected_neg += 1.

            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_non_protected += 1.
                else:
                    tn_non_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_non_protected += 1.
                else:
                    fp_non_protected += 1.

    tpr_protected = tp_protected / (tp_protected + fn_protected)
    tnr_protected = tn_protected / (tn_protected + fp_protected)

    tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
    tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)

    C_prot = (protected_pos) / (protected_pos + protected_neg)
    C_non_prot = (non_protected_pos) / (non_protected_pos + non_protected_neg)

    stat_par = C_non_prot - C_prot

    output = dict()

    # output["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
    output["balanced_accuracy"] =( (tp_protected + tp_non_protected)/(tp_protected + tp_non_protected + fn_protected + fn_non_protected) +
                                   (tn_protected + tn_non_protected) / (tn_protected + tn_non_protected + fp_protected + fp_non_protected))*0.5

    output["accuracy"] = accuracy_score(labels, predictions)
    # output["dTPR"] = tpr_non_protected - tpr_protected
    # output["dTNR"] = tnr_non_protected - tnr_protected
    output["fairness"] = abs(tpr_non_protected - tpr_protected) + abs(tnr_non_protected - tnr_protected)
    # output["fairness"] = abs(stat_par)

    output["TPR_protected"] = tpr_protected
    output["TPR_non_protected"] = tpr_non_protected
    output["TNR_protected"] = tnr_protected
    output["TNR_non_protected"] = tnr_non_protected
    return output

def calculate_performanceEQOP(data, labels, predictions, saIndex, saValue):
    protected_pos = 0.
    protected_neg = 0.
    non_protected_pos = 0.
    non_protected_neg = 0.

    tp_protected = 0.
    tn_protected = 0.
    fp_protected = 0.
    fn_protected = 0.

    tp_non_protected = 0.
    tn_non_protected = 0.
    fp_non_protected = 0.
    fn_non_protected = 0.
    for idx, val in enumerate(data):
        # protrcted population
        if val[saIndex] == saValue:
            if predictions[idx] == 1:
                protected_pos += 1.
            else:
                protected_neg += 1.

            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_protected += 1.
                else:
                    tn_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_protected += 1.
                else:
                    fp_protected += 1.

        else:
            if predictions[idx] == 1:
                non_protected_pos += 1.
            else:
                non_protected_neg += 1.

            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_non_protected += 1.
                else:
                    tn_non_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_non_protected += 1.
                else:
                    fp_non_protected += 1.

    tpr_protected = tp_protected / (tp_protected + fn_protected)
    tnr_protected = tn_protected / (tn_protected + fp_protected)

    tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
    tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)

    C_prot = (protected_pos) / (protected_pos + protected_neg)
    C_non_prot = (non_protected_pos) / (non_protected_pos + non_protected_neg)

    stat_par = C_non_prot - C_prot

    output = dict()

    # output["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
    output["balanced_accuracy"] =( (tp_protected + tp_non_protected)/(tp_protected + tp_non_protected + fn_protected + fn_non_protected) +
                                   (tn_protected + tn_non_protected) / (tn_protected + tn_non_protected + fp_protected + fp_non_protected))*0.5

    output["accuracy"] = accuracy_score(labels, predictions)
    # output["dTPR"] = tpr_non_protected - tpr_protected
    # output["dTNR"] = tnr_non_protected - tnr_protected
    output["fairness"] = abs(tpr_non_protected - tpr_protected)
    # output["fairness"] = abs(stat_par)

    output["TPR_protected"] = tpr_protected
    output["TPR_non_protected"] = tpr_non_protected
    output["TNR_protected"] = tnr_protected
    output["TNR_non_protected"] = tnr_non_protected
    return output


def calculate_performance_SP(data, labels, predictions, saIndex, saValue):
    protected_pos = 0.
    protected_neg = 0.
    non_protected_pos = 0.
    non_protected_neg = 0.

    tp_protected = 0.
    tn_protected = 0.
    fp_protected = 0.
    fn_protected = 0.

    tp_non_protected = 0.
    tn_non_protected = 0.
    fp_non_protected = 0.
    fn_non_protected = 0.

    for idx, val in enumerate(data):
        # protrcted population
        if val[saIndex] == saValue:
            if predictions[idx] == 1:
                protected_pos += 1.
            else:
                protected_neg += 1.
            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_protected += 1.
                else:
                    tn_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_protected += 1.
                else:
                    fp_protected += 1.

        else:
            if predictions[idx] == 1:
                non_protected_pos += 1.
            else:
                non_protected_neg += 1.

            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_non_protected += 1.
                else:
                    tn_non_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_non_protected += 1.
                else:
                    fp_non_protected += 1.

    tpr_protected = tp_protected / (tp_protected + fn_protected)
    tnr_protected = tn_protected / (tn_protected + fp_protected)

    tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
    tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)

    C_prot = (protected_pos) / (protected_pos + protected_neg)
    C_non_prot = (non_protected_pos) / (non_protected_pos + non_protected_neg)

    stat_par = C_non_prot - C_prot

    output = dict()

    # output["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
    output["balanced_accuracy"] =( (tp_protected + tp_non_protected)/(tp_protected + tp_non_protected + fn_protected + fn_non_protected) +
                                   (tn_protected + tn_non_protected) / (tn_protected + tn_non_protected + fp_protected + fp_non_protected))*0.5

    output["accuracy"] = accuracy_score(labels, predictions)
    #output["fairness"] = abs(stat_par)
    output["fairness"] = stat_par

    output["Positive_prot_pred"] = C_prot
    output["Positive_non_prot_pred"] = C_non_prot
    output["Negative_prot_pred"] = (protected_neg) / (protected_pos + protected_neg)
    output["Negative_non_prot_pred"] = (non_protected_neg) / (non_protected_pos + non_protected_neg)

    return output


def plot_results_of_c_impact_SP(csb1, csb2, steps, output_dir, dataset):
    csb1_accuracy_list = []
    csb1_balanced_accuracy_list = []
    csb1_fairness_list = []
    csb1_tpr_protected_list = []
    csb1_tpr_non_protected_list = []
    csb1_tnr_protected_list = []
    csb1_tnr_non_protected_list = []

    csb2_accuracy_list = []
    csb2_balanced_accuracy_list = []
    csb2_fairness_list = []
    csb2_tpr_protected_list = []
    csb2_tpr_non_protected_list = []
    csb2_tnr_protected_list = []
    csb2_tnr_non_protected_list = []

    for c in steps:

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        # for item in csb1[c]:
        #     accuracy.append(item["accuracy"])
        #     balanced_accuracy.append(item["balanced_accuracy"])
        #     fairness.append(item["fairness"])
        #     tpr_protected.append(item["Positive_prot_pred"])
        #     tpr_non_protected.append(item["Positive_non_prot_pred"])
        #     tnr_protected.append(item["Negative_prot_pred"])
        #     tnr_non_protected.append(item["Negative_non_prot_pred"])

        # csb1_tpr_protected_list.append(numpy.mean(tpr_protected))
        # csb1_tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        # csb1_tnr_protected_list.append(numpy.mean(tnr_protected))
        # csb1_tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in csb2[c]:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["Positive_prot_pred"])
            tpr_non_protected.append(item["Positive_non_prot_pred"])
            tnr_protected.append(item["Negative_prot_pred"])
            tnr_non_protected.append(item["Negative_non_prot_pred"])

        csb2_accuracy_list.append(numpy.mean(accuracy))
        csb2_balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        csb2_fairness_list.append(numpy.mean(fairness))
        csb2_tpr_protected_list.append(numpy.mean(tpr_protected))
        csb2_tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        csb2_tnr_protected_list.append(numpy.mean(tnr_protected))
        csb2_tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

    plt.figure(figsize=(5, 5))
    plt.grid(True)
    plt.rcParams.update({'font.size': 13})

    plt.plot(steps, csb2_accuracy_list, '--', label='Accuracy')

    plt.plot(steps, csb2_balanced_accuracy_list, '-*', label='Balanced Accuracy')

    plt.plot(steps, csb2_fairness_list, '-v', label='St. Parity')
    plt.plot(steps, csb2_tpr_protected_list, '-x', label='Prot. Pos.')
    plt.plot(steps, csb2_tpr_non_protected_list, '-o', label='Non-Prot. Pos.')
    plt.plot(steps, csb2_tnr_protected_list, '-^', label='Prot. Neg.')
    plt.plot(steps, csb2_tnr_non_protected_list, '-<', label='Non-Prot. Neg.')

    plt.legend(loc='center', bbox_to_anchor=(0.7, 0.3), shadow=False,ncol=1, framealpha=.30)

    plt.xlabel('c')
    # plt.ylabel('(%)')
    # plt.title("Impact of c for " + dataset + " dataset")

    print( "csb2_accuracy_list " + str(csb2_accuracy_list))
    print ("csb2_balanced_accuracy_list " + str(csb2_balanced_accuracy_list))
    print ("csb2_fairness_list " + str(csb2_fairness_list))
    print ("csb2_tpr_protected_list " + str(csb2_tpr_protected_list))
    print ("csb2_tpr_non_protected_list " + str(csb2_tpr_non_protected_list))
    print ("csb2_tnr_protected_list " + str(csb2_tnr_protected_list))
    print ("csb2_tnr_non_protected_list " + str(csb2_tnr_non_protected_list))

    plt.savefig(output_dir + dataset + "_sp_c_impact.png", bbox_inches='tight', dpi=200)



def plot_results_of_c_impact_EQOP(csb1, csb2, steps, output_dir, dataset):
    csb2_accuracy_list = []
    csb2_balanced_accuracy_list = []
    csb2_fairness_list = []
    csb2_tpr_protected_list = []
    csb2_tpr_non_protected_list = []
    csb2_tnr_protected_list = []
    csb2_tnr_non_protected_list = []

    for c in steps:
        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in csb2[c]:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["TPR_protected"])
            tpr_non_protected.append(item["TPR_non_protected"])
            tnr_protected.append(item["TNR_protected"])
            tnr_non_protected.append(item["TNR_non_protected"])

        csb2_accuracy_list.append(numpy.mean(accuracy))
        csb2_balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        csb2_fairness_list.append(numpy.mean(fairness))
        csb2_tpr_protected_list.append(numpy.mean(tpr_protected))
        csb2_tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        csb2_tnr_protected_list.append(numpy.mean(tnr_protected))
        csb2_tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

    plt.figure(figsize=(5, 5))
    plt.grid(True)
    plt.rcParams.update({'font.size': 13})

    plt.plot(steps, csb2_accuracy_list, '--', label='Accuracy')
    plt.plot(steps, csb2_balanced_accuracy_list, '-*', label='Balanced Accuracy')
    plt.plot(steps, csb2_fairness_list, '-v', label='Equal Opportunity')
    plt.plot(steps, csb2_tpr_protected_list, '-x', label='TPR Prot.')
    plt.plot(steps, csb2_tpr_non_protected_list, '-o', label='TPR Non-Prot.')
    plt.plot(steps, csb2_tnr_protected_list, '-^', label='TNR Prot.')
    plt.plot(steps, csb2_tnr_non_protected_list, '-<', label='TNR Non-Prot.')

    plt.legend(loc='center', bbox_to_anchor=(0.7, 0.3), shadow=False,ncol=1, framealpha=.30)

    plt.xlabel('c')
    # plt.ylabel('(%)')
    # plt.title("Impact of c for " + dataset + " dataset")

    print( "csb2_accuracy_list " + str(csb2_accuracy_list))
    print ("csb2_balanced_accuracy_list " + str(csb2_balanced_accuracy_list))
    print ("csb2_fairness_list " + str(csb2_fairness_list))
    print ("csb2_tpr_protected_list " + str(csb2_tpr_protected_list))
    print ("csb2_tpr_non_protected_list " + str(csb2_tpr_non_protected_list))
    print ("csb2_tnr_protected_list " + str(csb2_tnr_protected_list))
    print ("csb2_tnr_non_protected_list " + str(csb2_tnr_non_protected_list))

    plt.savefig(output_dir + dataset + "_eqop_c_impact.png", bbox_inches='tight', dpi=200)




def plot_results_of_c_impact(csb1, csb2, steps, output_dir, dataset):
    csb1_accuracy_list = []
    csb1_balanced_accuracy_list = []
    csb1_fairness_list = []
    csb1_tpr_protected_list = []
    csb1_tpr_non_protected_list = []
    csb1_tnr_protected_list = []
    csb1_tnr_non_protected_list = []

    csb2_accuracy_list = []
    csb2_balanced_accuracy_list = []
    csb2_fairness_list = []
    csb2_tpr_protected_list = []
    csb2_tpr_non_protected_list = []
    csb2_tnr_protected_list = []
    csb2_tnr_non_protected_list = []

    for c in steps:

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in csb1[c]:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["TPR_protected"])
            tpr_non_protected.append(item["TPR_non_protected"])
            tnr_protected.append(item["TNR_protected"])
            tnr_non_protected.append(item["TNR_non_protected"])

        csb1_accuracy_list.append(numpy.mean(accuracy))
        csb1_balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        csb1_fairness_list.append(numpy.mean(fairness))
        # csb1_tpr_protected_list.append(numpy.mean(tpr_protected))
        # csb1_tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        # csb1_tnr_protected_list.append(numpy.mean(tnr_protected))
        # csb1_tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in csb2[c]:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["TPR_protected"])
            tpr_non_protected.append(item["TPR_non_protected"])
            tnr_protected.append(item["TNR_protected"])
            tnr_non_protected.append(item["TNR_non_protected"])

        csb2_accuracy_list.append(numpy.mean(accuracy))
        csb2_balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        csb2_fairness_list.append(numpy.mean(fairness))
        csb2_tpr_protected_list.append(numpy.mean(tpr_protected))
        csb2_tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        csb2_tnr_protected_list.append(numpy.mean(tnr_protected))
        csb2_tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

    plt.figure(figsize=(5, 5))
    plt.grid(True)
    plt.rcParams.update({'font.size': 13})

    # plt.plot(steps, csb1_accuracy_list, '-.', label='AFB CSB1 Accuracy')
    plt.plot(steps, csb2_accuracy_list, '--', label='Accuracy')

    # plt.plot(steps, csb1_balanced_accuracy_list, '-x', label='AFB CSB1 B.Accuracy')
    plt.plot(steps, csb2_balanced_accuracy_list, '-*', label='Balanced Accuracy')

    # plt.plot(steps, csb1_fairness_list, '-o', label='AFB CSB1 E.O.')
    plt.plot(steps, csb2_fairness_list, '-v', label='Disp. Mis.')
    plt.plot(steps, csb2_tpr_protected_list, '-x', label='TPR Prot.')
    plt.plot(steps, csb2_tpr_non_protected_list, '-o', label='TPR Non-Prot.')
    plt.plot(steps, csb2_tnr_protected_list, '-^', label='TNR Prot.')
    plt.plot(steps, csb2_tnr_non_protected_list, '-<', label='TNR Non-Prot.')

    plt.legend(loc='center', bbox_to_anchor=(0.7, 0.3), shadow=False,ncol=1, framealpha=.30)

    plt.xlabel('c')
    # plt.ylabel('(%)')
    # plt.title("Impact of c for " + dataset + " dataset")

    print( "csb2_accuracy_list " + str(csb2_accuracy_list))
    print ("csb2_balanced_accuracy_list " + str(csb2_balanced_accuracy_list))
    print ("csb2_fairness_list " + str(csb2_fairness_list))
    print ("csb2_tpr_protected_list " + str(csb2_tpr_protected_list))
    print ("csb2_tpr_non_protected_list " + str(csb2_tpr_non_protected_list))
    print ("csb2_tnr_protected_list " + str(csb2_tnr_protected_list))
    print ("csb2_tnr_non_protected_list " + str(csb2_tnr_non_protected_list))

    plt.savefig(output_dir + dataset + "_dm_c_impact.png", bbox_inches='tight', dpi=200)

def plot_costs_per_round_eqop(output, noaccum, adafair):
    steps = [i for i in range(0, len(noaccum))]
    plt.figure(figsize=(5, 5))
    plt.grid(True)
    plt.rcParams.update({'font.size': 10.5})

    plt.plot(steps, noaccum, '-.', label='AdaFair NoCumul ' + r'$\delta$FNR')
    plt.plot(steps, adafair, '-', color='g', label='AdaFair ' + r'$\delta$FNR')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=2, shadow=False,fancybox=True, framealpha=1.0)
    plt.xlabel('Round')

    plt.savefig(output+ "_eqop_costs.png", bbox_inches='tight', dpi=200,shadow=False,fancybox=True, framealpha=.30)



def plot_costs_per_round_sp(output, noaccum, adafair):
    steps = [i for i in range(0, len(noaccum))]
    plt.figure(figsize=(5, 5))
    plt.grid(True)
    plt.rcParams.update({'font.size': 10.5})
    plt.ylim([-1,1])
    plt.plot(steps, noaccum, '-.', label='AdaFair NoCumul ' + r'$\delta$SP')
    plt.plot(steps, adafair, '-', color='g', label='AdaFair ' + r'$\delta$SP')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=2, shadow=False,fancybox=True, framealpha=1.0)
    plt.xlabel('Round')

    plt.savefig(output+ "_sp_costs.png", bbox_inches='tight', dpi=200,shadow=False,fancybox=True, framealpha=.30)

def plot_costs_per_round(output, noaccum, adafair):
    noaccum_tpr = []
    noaccum_tnr = []
    accum_tpr = []
    accum_tnr = []
    for i in noaccum:
        noaccum_tpr.append(float(i.split(",")[0]))
        noaccum_tnr.append(float(i.split(",")[1]))
    for i in adafair:
        accum_tpr.append(float(i.split(",")[0]))
        accum_tnr.append(float(i.split(",")[1]))

    steps = numpy.arange(0, len(accum_tnr), step=1)
    plt.figure(figsize=(5, 5))
    plt.grid(True)
    plt.rcParams.update({'font.size': 10.5})
    plt.ylim([-1,1])
    plt.plot(steps, noaccum_tpr, '-.', label='AdaFair NoCumul ' + r'$\delta$FNR')
    plt.plot(steps, noaccum_tnr, ':', label='AdaFair NoCumul ' + r'$\delta$FPR')
    plt.plot(steps, accum_tpr, '-', label='AdaFair ' + r'$\delta$FNR')
    plt.plot(steps, accum_tnr, '--', label='AdaFair ' + r'$\delta$FPR')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.105), ncol=2, shadow=False,fancybox=True, framealpha=1.0)

    # plt.legend(loc='best')

    plt.xlabel('Round')

    plt.savefig(output+ "_dm_costs.png", bbox_inches='tight', dpi=200,shadow=False,fancybox=True, framealpha=.30)


def plot_results(init, max_cost, step, summary_performance, summary_weights, output_dir, title, plot_weights=True):
    step_list = []
    accuracy_list = []
    auc_list = []
    average_precision_list = []
    d_tpr_list = []
    d_tnr_list = []
    tpr_protected_list = []
    tpr_non_protected_list = []
    tnr_protected_list = []
    tnr_non_protected_list = []
    #
    W_pos_list = []
    W_neg_list = []
    W_dp_list = []
    W_dn_list = []
    W_fp_list = []
    W_fn_list = []

    for i in numpy.arange(init, max_cost + step, step):

        iteration_performance_summary = summary_performance[i]
        iteration_weights_summary = summary_weights[i]
        W_pos = 0.
        W_neg = 0.
        W_dp = 0.
        W_dn = 0.
        W_fp = 0.
        W_fn = 0.

        accuracy = 0
        auc = 0
        average_precision = 0
        d_tpr = 0
        d_tnr = 0
        tpr_protected = 0
        tpr_non_protected = 0
        tnr_protected = 0
        tnr_non_protected = 0

        for item in iteration_weights_summary:
            W_pos += item[0] / len(iteration_weights_summary)
            W_neg += item[1] / len(iteration_weights_summary)
            W_dp += item[2] / len(iteration_weights_summary)
            W_fp += item[3] / len(iteration_weights_summary)
            W_dn += item[4] / len(iteration_weights_summary)
            W_fn += item[5] / len(iteration_weights_summary)

        W_pos_list.append(W_pos)
        W_neg_list.append(W_neg)
        W_dp_list.append(W_dp)
        W_fp_list.append(W_fp)
        W_dn_list.append(W_dn)
        W_fn_list.append(W_fn)

        for item in iteration_performance_summary:
            accuracy += item["accuracy"] / len(iteration_performance_summary)
            auc += item["auc"] / len(iteration_performance_summary)
            d_tpr += item["dTPR"] / len(iteration_performance_summary)
            d_tnr += item["dTNR"] / len(iteration_performance_summary)
            tpr_protected += item["TPR_protected"] / len(iteration_performance_summary)
            tpr_non_protected += item["TPR_non_protected"] / len(iteration_performance_summary)
            tnr_protected += item["TNR_protected"] / len(iteration_performance_summary)
            tnr_non_protected += item["TNR_non_protected"] / len(iteration_performance_summary)
            average_precision += item["average_precision"] / len(iteration_performance_summary)

        step_list.append(i)
        accuracy_list.append(accuracy)
        auc_list.append(auc)
        average_precision_list.append(average_precision)
        d_tpr_list.append(d_tpr)
        d_tnr_list.append(d_tnr)
        tpr_protected_list.append(tpr_protected)
        tpr_non_protected_list.append(tpr_non_protected)
        tnr_protected_list.append(tnr_protected)
        tnr_non_protected_list.append(tnr_non_protected)

    plt.figure(figsize=(20, 20))
    plt.grid(True)

    plt.plot(step_list, accuracy_list, '-b', label='accuracy')
    plt.plot(step_list, auc_list, '-r', label='auc')

    plt.plot(step_list, average_precision_list, '-*', label='average precision')

    plt.plot(step_list, d_tpr_list, '--', label='dTPR')
    plt.plot(step_list, d_tnr_list, ':', label='dTNR')

    plt.plot(step_list, tpr_protected_list, '-o', label='TPR Prot.')
    plt.plot(step_list, tpr_non_protected_list, '-v', label='TPR non-Prot.')

    plt.plot(step_list, tnr_protected_list, '-.', label='TNR Prot.')
    plt.plot(step_list, tnr_non_protected_list, '-+', label='TNR non-Prot.')
    plt.legend(loc='best')

    plt.xlabel('Cost Increasement (%)')
    plt.ylabel('(%)')
    plt.title(title + " performance")

    plt.savefig(output_dir + "_performance.png")

    if not plot_weights:
        return

    plt.figure(figsize=(20, 20))
    plt.grid(True)

    plt.plot(step_list, W_pos_list, '-b', label='Positives')
    plt.plot(step_list, W_neg_list, '-r', label='Negatives')

    plt.plot(step_list, W_dp_list, '--', label='Prot. Positives')
    plt.plot(step_list, W_fp_list, ':', label='Non-Prot. Positives')

    plt.plot(step_list, W_dn_list, '-o', label='Prot. Negatives')
    plt.plot(step_list, W_fn_list, '-+', label='Non-Prot. Negatives')
    plt.legend(loc='best')

    # plt.legend(loc='best')

    plt.xlabel('Cost Increasement (%)')
    plt.ylabel('(%)')
    plt.title(title + " weights")

    plt.savefig(output_dir + "_weights.png")
    # plt.show()


def plot_per_round(rounds, results, objective, output_dir ):
    train_error_list = []
    train_bal_error_list = []
    train_fairness = []

    test_error_list = []
    test_bal_error_list = []
    test_fairness = []
    objective_list = []
    for i in numpy.arange(0, rounds):

        line = results[i].split(",")

        objective_list.append(objective[i])
        train_error_list.append(float(line[1]))
        train_bal_error_list.append(float(line[2]))
        train_fairness.append(float(line[3]))
        test_error_list.append(float(line[4]))
        test_bal_error_list.append(float(line[5]))
        test_fairness.append(float(line[6]))

    step_list = [i for i in range(0, rounds)]

    print (train_fairness)
    print (train_error_list)
    print (train_bal_error_list)
    print (step_list)
    plt.figure()
    plt.grid(True)

    plt.plot(step_list, train_error_list, '--', label='Train Error rate')
    plt.plot(step_list, test_error_list, ':', label='Test Error rate')

    plt.plot(step_list, train_bal_error_list, '-.', label='Train Bal.Error rate')
    plt.plot(step_list, test_bal_error_list, '-', label='Test Bal.Error rate')

    plt.plot(step_list, train_fairness, '-x', label='Train E.O.', markersize=3.25)
    plt.plot(step_list, objective_list, '-<', label='Objective', markersize=3.25)
    plt.plot(step_list, test_fairness, '-o', label='Test E.O.', markersize=3.25)

    # plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 10})
    # plt.ylim([0,1])
    # plt.yticks(numpy.arange(0, 1.00001, step=0.05))

    plt.xlabel('Rounds')
    plt.ylabel('(%)')
    # plt.legend(loc='best',ncol=1, shadow=False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=3, shadow=False,fancybox=True, framealpha=1.0)

    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir,bbox_inches='tight', dpi=200)



def plot_my_results(results, names, output_dir, dataset):
    accuracy_list = []
    balanced_accuracy_list = []
    fairness_list = []
    tpr_protected_list = []
    tpr_non_protected_list = []
    tnr_protected_list = []
    tnr_non_protected_list = []
    std_accuracy_list = []
    std_balanced_accuracy_list = []
    std_fairness_list = []
    std_tpr_protected_list = []
    std_tpr_non_protected_list = []
    std_tnr_protected_list = []
    std_tnr_non_protected_list = []

    for list_of_results in results:

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in list_of_results:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["TPR_protected"])
            tpr_non_protected.append(item["TPR_non_protected"])
            tnr_protected.append(item["TNR_protected"])
            tnr_non_protected.append(item["TNR_non_protected"])

        numpy.mean(accuracy)
        numpy.std(accuracy)

        accuracy_list.append(numpy.mean(accuracy))
        balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        fairness_list.append(numpy.mean(fairness))
        tpr_protected_list.append(numpy.mean(tpr_protected))
        tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        tnr_protected_list.append(numpy.mean(tnr_protected))
        tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

        std_accuracy_list.append(numpy.std(accuracy))
        std_balanced_accuracy_list.append(numpy.std(balanced_accuracy))
        std_fairness_list.append(numpy.std(fairness))
        std_tpr_protected_list.append(numpy.std(tpr_protected))
        std_tpr_non_protected_list.append(numpy.std(tpr_non_protected))
        std_tnr_protected_list.append(numpy.std(tnr_protected))
        std_tnr_non_protected_list.append(numpy.std(tnr_non_protected))

    plt.figure(figsize=(9, 9))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 8, step=1.3)
    # index = numpy.arange(7)
    bar_width = 0.175



    plt.xticks(index + 1.5*bar_width ,
               ('Accuracy', 'Balanced Accuracy', 'Disp. Mis.', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))

    colors = ['b','g','r','c','m','y','k', 'dimgray']
    for i in range(0, len(names)):
        plt.bar(index + bar_width * i,
                [accuracy_list[i], balanced_accuracy_list[i], fairness_list[i], tpr_protected_list[i],
                 tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
                yerr=[std_accuracy_list[i], std_balanced_accuracy_list[i], std_fairness_list[i],
                      std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]],
                label=names[i], color=colors[i],edgecolor='black')

    plt.legend(loc='best',ncol=1, shadow=False)
    plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir + "_performance.png",bbox_inches='tight', dpi=200)
    print (names)
    print ("accuracy_list= " + str(accuracy_list))
    print ("std_accuracy_list = " + str(std_accuracy_list))
    print ("balanced_accuracy_list=  " + str(balanced_accuracy_list))
    print ("std_balanced_accuracy_list = " + str(std_balanced_accuracy_list))

    print ("fairness_list=  " + str(fairness_list))
    print ("std_fairness_list = " + str(std_fairness_list))

    print ("tpr_protected_list = " + str(tpr_protected_list))
    print ("std_tpr_protected_list = " + str(std_tpr_protected_list))

    print ("tpr_non_protected_list = " + str(tpr_non_protected_list))
    print ("std_tpr_non_protected_list = " + str(std_tpr_non_protected_list))

    print ("tnr_protected_list = " + str(tnr_protected_list))
    print ("std_tnr_protected_list = " + str(std_tnr_protected_list))

    print ("tnr_non_protected_list = " + str(tnr_non_protected_list))
    print ("std_tnr_non_protected_list = " + str(std_tnr_non_protected_list))


def plot_my_resultsEQOP(results, names, output_dir, dataset):
    accuracy_list = []
    balanced_accuracy_list = []
    fairness_list = []
    tpr_protected_list = []
    tpr_non_protected_list = []
    tnr_protected_list = []
    tnr_non_protected_list = []
    std_accuracy_list = []
    std_balanced_accuracy_list = []
    std_fairness_list = []
    std_tpr_protected_list = []
    std_tpr_non_protected_list = []
    std_tnr_protected_list = []
    std_tnr_non_protected_list = []

    for list_of_results in results:

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in list_of_results:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["TPR_protected"])
            tpr_non_protected.append(item["TPR_non_protected"])
            tnr_protected.append(item["TNR_protected"])
            tnr_non_protected.append(item["TNR_non_protected"])

        numpy.mean(accuracy)
        numpy.std(accuracy)

        accuracy_list.append(numpy.mean(accuracy))
        balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        fairness_list.append(numpy.mean(fairness))
        tpr_protected_list.append(numpy.mean(tpr_protected))
        tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        tnr_protected_list.append(numpy.mean(tnr_protected))
        tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

        std_accuracy_list.append(numpy.std(accuracy))
        std_balanced_accuracy_list.append(numpy.std(balanced_accuracy))
        std_fairness_list.append(numpy.std(fairness))
        std_tpr_protected_list.append(numpy.std(tpr_protected))
        std_tpr_non_protected_list.append(numpy.std(tpr_non_protected))
        std_tnr_protected_list.append(numpy.std(tnr_protected))
        std_tnr_non_protected_list.append(numpy.std(tnr_non_protected))

    #
    plt.figure(figsize=(9, 9))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 8, step=1.3)
    # index = numpy.arange(7)
    bar_width = 0.175



    plt.xticks(index + 1.5*bar_width ,
               ('Accuracy', 'Balanced Accuracy', 'Equal Opportunity', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))

    colors = ['b','g','r','c','m','y','k', 'dimgray']
    for i in range(0, len(names)):
        plt.bar(index + bar_width * i,
                [accuracy_list[i], balanced_accuracy_list[i], fairness_list[i], tpr_protected_list[i],
                 tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
                yerr=[std_accuracy_list[i], std_balanced_accuracy_list[i], std_fairness_list[i],
                      std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]],
                label=names[i], color=colors[i],edgecolor='black')

    plt.legend(loc='best',ncol=1, shadow=False)
    plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir + "_performance.png",bbox_inches='tight', dpi=200)
    print (names)
    print ("accuracy_list= " + str(accuracy_list))
    print ("std_accuracy_list = " + str(std_accuracy_list))
    print ("balanced_accuracy_list=  " + str(balanced_accuracy_list))
    print ("std_balanced_accuracy_list = " + str(std_balanced_accuracy_list))

    print ("fairness_list=  " + str(fairness_list))
    print ("std_fairness_list = " + str(std_fairness_list))

    print ("tpr_protected_list = " + str(tpr_protected_list))
    print ("std_tpr_protected_list = " + str(std_tpr_protected_list))

    print ("tpr_non_protected_list = " + str(tpr_non_protected_list))
    print ("std_tpr_non_protected_list = " + str(std_tpr_non_protected_list))

    print ("tnr_protected_list = " + str(tnr_protected_list))
    print ("std_tnr_protected_list = " + str(std_tnr_protected_list))

    print ("tnr_non_protected_list = " + str(tnr_non_protected_list))
    print ("std_tnr_non_protected_list = " + str(std_tnr_non_protected_list))


def plot_my_results_sp(results, names, output_dir, dataset):
    accuracy_list = []
    balanced_accuracy_list = []
    fairness_list = []
    tpr_protected_list = []
    tpr_non_protected_list = []
    tnr_protected_list = []
    tnr_non_protected_list = []
    std_accuracy_list = []
    std_balanced_accuracy_list = []
    std_fairness_list = []
    std_tpr_protected_list = []
    std_tpr_non_protected_list = []
    std_tnr_protected_list = []
    std_tnr_non_protected_list = []

    for list_of_results in results:

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in list_of_results:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["Positive_prot_pred"])
            tpr_non_protected.append(item["Positive_non_prot_pred"])
            tnr_protected.append(item["Negative_prot_pred"])
            tnr_non_protected.append(item["Negative_non_prot_pred"])

        numpy.mean(accuracy)
        numpy.std(accuracy)

        accuracy_list.append(numpy.mean(accuracy))
        balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        fairness_list.append(numpy.mean(fairness))
        tpr_protected_list.append(numpy.mean(tpr_protected))
        tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        tnr_protected_list.append(numpy.mean(tnr_protected))
        tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

        std_accuracy_list.append(numpy.std(accuracy))
        std_balanced_accuracy_list.append(numpy.std(balanced_accuracy))
        std_fairness_list.append(numpy.std(fairness))
        std_tpr_protected_list.append(numpy.std(tpr_protected))
        std_tpr_non_protected_list.append(numpy.std(tpr_non_protected))
        std_tnr_protected_list.append(numpy.std(tnr_protected))
        std_tnr_non_protected_list.append(numpy.std(tnr_non_protected))

    plt.figure(figsize=(9, 9))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 8, step=1.3)
    # index = numpy.arange(7)
    bar_width = 0.175



    plt.xticks(index + 1.5*bar_width ,
               ('Accuracy', 'Balanced Accuracy', 'St.Parity', 'Pos.Prot', 'Pos.Non-Prot.', 'Neg.Prot.', 'Neg.Non-Prot.'))

    colors = ['b','g','r','c','m','y','k', 'dimgray']
    for i in range(0, len(names)):
        plt.bar(index + bar_width * i,
                [accuracy_list[i], balanced_accuracy_list[i], fairness_list[i], tpr_protected_list[i],
                 tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
                yerr=[std_accuracy_list[i], std_balanced_accuracy_list[i], std_fairness_list[i],
                      std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]],
                label=names[i], color=colors[i],edgecolor='black')

    plt.legend(loc='best',ncol=1, shadow=False)
    plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir + "_performance.png",bbox_inches='tight', dpi=200)
    print (names)
    print ("accuracy_list= " + str(accuracy_list))
    print ("std_accuracy_list = " + str(std_accuracy_list))
    print ("balanced_accuracy_list=  " + str(balanced_accuracy_list))
    print ("std_balanced_accuracy_list = " + str(std_balanced_accuracy_list))

    print ("fairness_list=  " + str(fairness_list))
    print ("std_fairness_list = " + str(std_fairness_list))

    print ("tpr_protected_list = " + str(tpr_protected_list))
    print ("std_tpr_protected_list = " + str(std_tpr_protected_list))

    print ("tpr_non_protected_list = " + str(tpr_non_protected_list))
    print ("std_tpr_non_protected_list = " + str(std_tpr_non_protected_list))

    print ("tnr_protected_list = " + str(tnr_protected_list))
    print ("std_tnr_protected_list = " + str(std_tnr_protected_list))

    print ("tnr_non_protected_list = " + str(tnr_non_protected_list))
    print ("std_tnr_non_protected_list = " + str(std_tnr_non_protected_list))



def plot_calibration_curves(results, names, init_cost, max_cost, step, directory):
    for num in range(init_cost, max_cost + step, step):
        plt.figure(figsize=(10, 10))
        # ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        # ax2 = plt.subplot2grid((3, 1), (2, 0))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        for idx, row in enumerate(results):
            plt.plot(map(mean, zip(*row.mean_predicted_value[num])), map(mean, zip(*row.fraction_of_positives[num])),
                     "s-", label="%s" % (names[idx][1:],))

        plt.ylabel("Fraction of positives")
        plt.legend(loc="best")
        plt.title('Calibration plots  (reliability curve) for cost = ' + str(num))
        plt.savefig(directory + "calibration_cost_" + str(num) + ".png")
        plt.show()


def mean(a):
    return sum(a) / len(a)




def plot_my_results_from_list_sp(dataset):
    names = ['Adaboost', 'AdaFair', 'SMOTEBoost', 'Adaboost SDB']
    output_dir = "Images/StatisticalParity/" + dataset
    if dataset == 'adult-gender':
        accuracy_list= [0.85125730476359129, 0.773663006906322, 0.81124048167168417, 0.78702851071365321]
        std_accuracy_list = [0.0021313582417819251, 0.015662871786919337, 0.0044718850717888311, 0.0070316474865371502]
        balanced_accuracy_list=  [0.76581541065707015, 0.7787438485338048, 0.81655255038861085, 0.71928209156998302]
        std_balanced_accuracy_list = [0.0034165442777111987, 0.0024571762875774377, 0.0020128690181906088, 0.0032339196570538185]
        fairness_list=  [0.20265716335167369, 0.007382628552249054, 0.35570869005043682, 0.010809828341700727]
        std_fairness_list = [0.0058196319523154804, 0.0071724489797373985, 0.011098563528078658, 0.0096481420749177267]
        tpr_protected_list = [0.059635144546631394, 0.3701746376219967, 0.1108554981243122, 0.25289675750534757]
        std_tpr_protected_list = [0.0037548904955061178, 0.0323677711507874, 0.0063990508875998017, 0.020030758560065903]
        tpr_non_protected_list = [0.26229230789830504, 0.3692727213870132, 0.46656418817474893, 0.25609558609262623]
        std_tpr_non_protected_list = [0.0055284524593893344, 0.029249663940729944, 0.010109904658918398, 0.0084334762746968907]
        tnr_protected_list = [0.94036485545336868, 0.6298253623780032, 0.88914450187568783, 0.74710324249465243]
        std_tnr_protected_list = [0.0037548904955061308, 0.03236777115078741, 0.0063990508875998182, 0.02003075856006592]
        tnr_non_protected_list = [0.73770769210169507, 0.6307272786129867, 0.53343581182525102, 0.74390441390737383]
        std_tnr_non_protected_list = [0.0055284524593893127, 0.029249663940729934, 0.0101099046589184, 0.0084334762746968942]
    elif dataset == 'compass-gender':
        accuracy_list = [0.6669571807502843, 0.650587343690792, 0.6555134520651762, 0.6362637362637363]
        std_accuracy_list = [0.007792455261771766, 0.009756214576566535, 0.008375070057082604, 0.012122785243482398]
        balanced_accuracy_list = [0.66434455862629, 0.6531187815188607, 0.6591398284622676, 0.6348890140929784]
        std_balanced_accuracy_list = [0.00782117593313684, 0.00808830121087754, 0.007718503992418956,
                                      0.01115195416510007]
        fairness_list = [0.24070382315099592, 0.0191420024337638, 0.27431136362679287, 0.06274669116530895]
        std_fairness_list = [0.08598787929008946, 0.011041153634911746, 0.06244729490547484, 0.04345495906672036]
        tpr_protected_list = [0.24991696918624723, 0.5287389716037609, 0.3274355449292631, 0.44094699054053593]
        std_tpr_protected_list = [0.05254444926242508, 0.059562211779729674, 0.06586283772387838, 0.0935118466354286]
        tpr_non_protected_list = [0.4906207923372432, 0.5350102704640852, 0.6017469085560558, 0.46805240456244135]
        std_tpr_non_protected_list = [0.04284733779303079, 0.05910951434609068, 0.027993819650889203,
                                      0.03828690920323129]
        tnr_protected_list = [0.7500830308137527, 0.4712610283962391, 0.672564455070737, 0.5590530094594641]
        std_tnr_protected_list = [0.052544449262425096, 0.05956221177972967, 0.06586283772387841, 0.09351184663542858]
        tnr_non_protected_list = [0.5093792076627568, 0.4649897295359148, 0.39825309144394405, 0.5319475954375587]
        std_tnr_non_protected_list = [0.04284733779303078, 0.05910951434609069, 0.02799381965088921, 0.03828690920323129]

    elif dataset == 'bank':
        accuracy_list= [0.9000949905009501, 0.8969103089691031, 0.87489251074892516, 0.86965803419658039]
        std_accuracy_list = [0.0011923676604366244, 0.0018050895539437693, 0.0042044695634957732, 0.0099246170913611549]
        balanced_accuracy_list=  [0.66210860705745578, 0.7438336575212308, 0.78948654025519849, 0.64137591134168526]
        std_balanced_accuracy_list = [0.0049149868796822442, 0.008853446174547372, 0.0069193678371777517, 0.010500225706748996]
        fairness_list=  [0.041691949042215071, 0.004035701965156792, 0.089660782964527647, 0.0070017681591029697]
        std_fairness_list = [0.0035676910001974506, 0.00208709158876769, 0.012924732000274877, 0.003516998439394659]
        tpr_protected_list = [0.052159906600296069, 0.10712181752931135, 0.13796930804424917, 0.094321523199343282]
        std_tpr_protected_list = [0.0023387056304894805, 0.006726767823488218, 0.0076966352989901619, 0.01234879957535282]
        tpr_non_protected_list = [0.093851855642511126, 0.10993577711975958, 0.22763009100877682, 0.093302239414308277]
        std_tpr_non_protected_list = [0.0032120931541992619, 0.005594537382633125, 0.015127999360102978, 0.0073548140987078686]
        tnr_protected_list = [0.94784009339970388, 0.8928781824706886, 0.86203069195575088, 0.90567847680065672]
        std_tnr_protected_list = [0.0023387056304894653, 0.006726767823488231, 0.007696635298990181, 0.012348799575352823]
        tnr_non_protected_list = [0.90614814435748881, 0.8900642228802405, 0.77236990899122326, 0.90669776058569163]
        std_tnr_non_protected_list = [0.0032120931541992619, 0.00559453738263314, 0.015127999360102992, 0.0073548140987078686]
    elif dataset == 'kdd':
        accuracy_list= [0.95096930695054238, 0.9335017341272229, 0.94385504166583145, 0.91699645155470011]
        std_accuracy_list = [0.0003205607874917051, 0.0019643190379017767, 0.00093095700033596991, 0.035481951218315315]
        balanced_accuracy_list=  [0.67010511019216046, 0.7520015372257142, 0.76884279202169958, 0.65380465799868015]
        std_balanced_accuracy_list = [0.0028052829061364568, 0.00901769434497467, 0.0047564858286854323, 0.017238385852276938]
        fairness_list=  [0.052004541336407396, 0.0023523142912902214, 0.095038001905124098, 0.040468017364132837]
        std_fairness_list = [0.0019092159564122126, 0.0013378404666003984, 0.0025780642771537207, 0.055251555925834618]
        tpr_protected_list = [0.0053969736450357204, 0.08876529540074965, 0.019211474230502793, 0.070024979762886394]
        std_tpr_protected_list = [0.00033560066736028452, 0.004329087764614954, 0.0016359101969885092, 0.068341732220019519]
        tpr_non_protected_list = [0.057401514981443115, 0.09008682248694556, 0.11424947613562689, 0.059031899618100805]
        std_tpr_non_protected_list = [0.0017738264522089721, 0.0046730753826213535, 0.0031806117445401889, 0.0016631601051075779]
        tnr_protected_list = [0.99460302635496434, 0.9112347045992503, 0.98078852576949715, 0.92997502023711365]
        std_tnr_protected_list = [0.00033560066736028522, 0.004329087764614965, 0.0016359101969884921, 0.068341732220019533]
        tnr_non_protected_list = [0.94259848501855692, 0.9099131775130544, 0.88575052386437325, 0.94096810038189904]
        std_tnr_non_protected_list = [0.0017738264522089895, 0.004673075382621363, 0.0031806117445401707, 0.0016631601051075957]

    plt.figure(figsize=(9, 9))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1.01, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 8, step=1.3)
    # index = numpy.arange(7)
    bar_width = 0.175



    # plt.xticks(index + 1.5*bar_width , ('Accuracy', 'Balanced Accuracy', 'Equalized Odds', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))
    plt.xticks(index + 1.5*bar_width , ('Accuracy', 'Bal. Acc.', 'St.Parity', 'Prot. Pos.', 'Non-Prot. Pos.', 'Prot. Neg.', 'Non-Prot. Neg.'))



    colors = [  "#B0B0B0","#E74C3C","#3498DB","#2ECC71"]
    # for i in range(0, len(names)):
    plt.bar(index + bar_width * 0,
            [accuracy_list[3], balanced_accuracy_list[3], fairness_list[3], tpr_protected_list[3],tpr_non_protected_list[3], tnr_protected_list[3], tnr_non_protected_list[3]], bar_width,
            yerr=[std_accuracy_list[3], std_balanced_accuracy_list[3], std_fairness_list[3], std_tpr_protected_list[3], std_tpr_non_protected_list[3], std_tnr_protected_list[3],
                  std_tnr_non_protected_list[3]], label=names[3], color=colors[2],edgecolor='black')


    plt.bar(index + bar_width * 1,
            [accuracy_list[0], balanced_accuracy_list[0], fairness_list[0], tpr_protected_list[0],tpr_non_protected_list[0], tnr_protected_list[0], tnr_non_protected_list[0]], bar_width,
            yerr=[std_accuracy_list[0], std_balanced_accuracy_list[0], std_fairness_list[0], std_tpr_protected_list[0], std_tpr_non_protected_list[1], std_tnr_protected_list[0],
                  std_tnr_non_protected_list[0]], label=names[0], color=colors[0], edgecolor='black')

    plt.bar(index + bar_width * 2,
            [accuracy_list[2], balanced_accuracy_list[2], fairness_list[2], tpr_protected_list[2],tpr_non_protected_list[2], tnr_protected_list[2], tnr_non_protected_list[2]], bar_width,
            yerr=[std_accuracy_list[2], std_balanced_accuracy_list[2], std_fairness_list[2], std_tpr_protected_list[2], std_tpr_non_protected_list[2], std_tnr_protected_list[2],
                  std_tnr_non_protected_list[2]], label=names[2], color=colors[1],edgecolor='black')

    plt.bar(index + bar_width * 3,
            [accuracy_list[1], balanced_accuracy_list[1], fairness_list[1], tpr_protected_list[1],tpr_non_protected_list[1], tnr_protected_list[1], tnr_non_protected_list[1]], bar_width,
            yerr=[std_accuracy_list[1], std_balanced_accuracy_list[1], std_fairness_list[1], std_tpr_protected_list[1], std_tpr_non_protected_list[1], std_tnr_protected_list[1],
                  std_tnr_non_protected_list[1]], label=names[1], color=colors[3],edgecolor='black')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=4, shadow=False,fancybox=True, framealpha=1.0)
    # plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir + "_sp_performance.png",bbox_inches='tight', dpi=200)


def plot_my_results_from_list_eqop(dataset):
    names = ['Adaboost', 'AdaFair', 'SMOTEBoost', 'FAE']
    output_dir = "Images/EqualOpportunity/" + dataset
    
    if dataset == 'adult-gender':
        accuracy_list= [0.85125730476359129,0.785102709403067, 0.81360014166814243, 0.7718909107008456 ]
        std_accuracy_list = [0.0021313582417819251, 0.0145866417927737667, 0.0037303235479497101, 0.011879299630220608 ]
        balanced_accuracy_list=  [0.76581541065707015, 0.8123122881905, 0.81621455272353227, 0.7923158733406547]
        std_balanced_accuracy_list = [0.0034165442777111987, 00.002835221496869, 0.0017686647761703674, 0.002826012688138698]
        fairness_list=  [0.19235516797286589,.01668140279819440502, 0.23200711712535602, 0.020733687320755933]
        std_fairness_list = [0.030051804326223008,  0.009142832241339124, 0.031476138004416419, 0.014441999118292375]
        tpr_protected_list = [0.43238015524640588,  0.8646042886302247998, 0.62377106310226216, 0.820023359117813]
        std_tpr_protected_list = [0.026993921544918097, 0.029097729714, 0.030470375545631919, 0.025738981830001263]
        tpr_non_protected_list = [0.62473532321927183, 00.866577030071302358, 0.8557781802276182, 0.8345630805321845]
        std_tpr_non_protected_list = [0.0096463265823206277, .02743501031574507302, 0.0087798744757660308,  0.027276438672765704]
        tnr_protected_list = [0.98796080170609046, .8451507638116822376, 0.95724707529353714, 0.8670520169331951]
        std_tnr_protected_list = [0.0013381003412159241, 0.0232457876243829, 0.0056165132873236814, 0.02096473014822554]
        tnr_non_protected_list = [0.90283444559215265, .72437176195692923, 0.72027790575674677, 0.680980527646855]
        std_tnr_non_protected_list = [0.0051467651774054603, 00.0317936763847606195, 0.010260785062740344,  0.027085604926498946]
    elif dataset == 'compass-gender':
        accuracy_list = [0.6669571807502843, .64910951117, 0.656309208033346, 0.6360363774156877]
        std_accuracy_list = [0.007792455261771766,  0.019728466411892, 0.010757915548752078, 0.012350199687557593]
        balanced_accuracy_list = [0.66434455862629, .65177508436059, 0.6600827079741186, 0.6401312277291022]
        std_balanced_accuracy_list = [0.00782117593313684, 0.0150174663251441657, 0.009502593227186553,0.010465191868153471]
        fairness_list = [0.22694551709870803, .02785231316522717, 0.247965671464255, 0.04563401185418984]
        std_fairness_list = [0.10244441374238589, .02946108064496136, 0.08798679197871552, 0.03817203788894951]
        tpr_protected_list = [0.4253301570360016, .69177501326, 0.509480150113111, 0.6956417052009246]
        std_tpr_protected_list = [0.0720115192336582, 0.06517762440619144, 0.07924523659496632, 0.032093212799435625]
        tpr_non_protected_list = [0.6522756741347097, 0.69730322293205, 0.7574458215773658, 0.7444757732810334]
        std_tpr_non_protected_list = [0.04393501872888887, 0.0769649769927925, 0.01886079474549104, 0.06259978132857417]
        tnr_protected_list = [0.8468457980288374, 0.56160071748830119, 0.7798719163393386, 0.5589835780717275]
        std_tnr_protected_list = [0.04871587937598028, 0.0883759360468471606, 0.05513476973961929, 0.04510550598840645]
        tnr_non_protected_list = [0.668500480234014, 0.6206982395553183, 0.5443889322665558, .5606504997761336]
        std_tnr_non_protected_list = [0.046849064790145126, 0.085651285287601484, 0.0381608182174372, 0.07623538135080789]
    elif dataset == 'bank':
        accuracy_list= [0.90009499050094988, 0.886616338514258, 0.87098790120987901, 0.7931806819318069]
        std_accuracy_list = [0.0011923676604366244,0.00976095122406071, 0.0050694166277808405, 0.006075196953787954]
        balanced_accuracy_list=  [0.66210860705745567, 0.79401921014, 0.79265309263554562, 0.8076725434153347]
        std_balanced_accuracy_list = [0.0049149868796822442, 0.021187025211562281, 0.0050568728357864683, 0.00532987182467402]
        fairness_list=  [0.082409417988262151, 0.01928691698796191, 0.094338491850008172, 0.015902230433248708]
        std_fairness_list = [0.01831911437453038, 0.013819733007404186, 0.033629513836561771, 0.0094484624130022  ]
        tpr_protected_list = [0.31811771871163785, 0.64668047, 0.65180304227645969, 0.8267025641738457]
        std_tpr_protected_list = [0.014701760130666241, 0.069791705499161388, 0.016088669221312787, 0.014143152400590156]
        tpr_non_protected_list = [0.40052713669989998, 0.6599, 0.74614153412646789, 0.816543043101094]
        std_tpr_non_protected_list = [0.013184925292930135, 0.07352417411490916702, 0.031227957465693647, 0.00533287645748039]
        tnr_protected_list = [0.97753569792411787, 0.9249999, 0.91594590560276856, 0.7528659813645353]
        std_tnr_protected_list = [0.001578348457488543, 0.01825211, 0.0070801621407839662, 0.014711497242106588]
        tnr_non_protected_list = [0.96022120901047203, 0.901763123, 0.84694648329272704, 0.8047093092757714]
        std_tnr_non_protected_list = [0.0022196508400848152, 0.022612231, 0.019912565390148657, 0.006487498149501307]
    elif dataset == 'kdd':
        accuracy_list= [0.95096930695054227, 0.9010605, 0.94350687970703606, 0.7760294569706364]
        std_accuracy_list = [0.0003205607874917051, 0.0123120237980150084309, 0.00072928745870637574, 0.017381877219775286]
        balanced_accuracy_list=  [0.67010511019216046, 0.84039189925026559, 0.76862451179642632, 0.8309845441564062]
        std_balanced_accuracy_list = [0.0028052829061364568, 0.0106182017362436847, 0.004180099571768083, 0.005019994609921996]
        fairness_list=  [0.27457006162494474, 0.00811362278752594369, 0.32327153473224479, 0.007583720507963007]
        std_fairness_list = [0.012853469843516568, 0.00363390743780346523, 0.016223839691663668, 0.012530594812777545]
        tpr_protected_list = [0.1337918437742191, 0.7673145275168526, 0.31511600591705491, 0.8937369305358157]
        std_tpr_protected_list = [0.0060130875772270043, 0.038543168206689258, 0.014749705252314315, 0.014196401225727207]
        tpr_non_protected_list = [0.40836190539916384, 0.77352144507729271, 0.6383875406492997, 0.89639924390503]
        std_tpr_non_protected_list = [0.0084998153592860202, 0.036446819123051484, 0.010881377126951641, 0.012723612525953913]
        tnr_protected_list = [0.9979799508908197, 0.91543278365979992, 0.98779847070972671, 0.7696201902968531]
        std_tnr_protected_list = [0.00027043660848440748, 0.0123189966275389258, 0.0010913981570608276, 0.024717910290754954]
        tnr_non_protected_list = [0.98243238104616726, 0.902642894824884225, 0.94535996657650789, 0.7663731181929454]
        std_tnr_non_protected_list = [0.0012111197845660435, 0.01673489204307783, 0.0020289538389142094,  0.014998042842229355]

    plt.figure(figsize=(9, 9))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1.01, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 8, step=1.3)
    # index = numpy.arange(7)
    bar_width = 0.175



    plt.xticks(index + 1.5*bar_width , ('Accuracy', 'Bal. Acc.', 'Eq. Op.', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))

    colors = [  "#B0B0B0","#E74C3C","#3498DB","#2ECC71"]


    plt.bar(index + bar_width * 0,
            [accuracy_list[3], balanced_accuracy_list[3], fairness_list[3], tpr_protected_list[3],tpr_non_protected_list[3], tnr_protected_list[3], tnr_non_protected_list[3]], bar_width,
            yerr=[std_accuracy_list[3], std_balanced_accuracy_list[3], std_fairness_list[3], std_tpr_protected_list[3], std_tpr_non_protected_list[3], std_tnr_protected_list[3],
                  std_tnr_non_protected_list[3]], label=names[3], color=colors[2],edgecolor='black')


    plt.bar(index + bar_width * 1,
            [accuracy_list[0], balanced_accuracy_list[0], fairness_list[0], tpr_protected_list[0],tpr_non_protected_list[0], tnr_protected_list[0], tnr_non_protected_list[0]], bar_width,
            yerr=[std_accuracy_list[0], std_balanced_accuracy_list[0], std_fairness_list[0], std_tpr_protected_list[0], std_tpr_non_protected_list[1], std_tnr_protected_list[0],
                  std_tnr_non_protected_list[0]], label=names[0], color=colors[0], edgecolor='black')

    plt.bar(index + bar_width * 2,
            [accuracy_list[2], balanced_accuracy_list[2], fairness_list[2], tpr_protected_list[2],tpr_non_protected_list[2], tnr_protected_list[2], tnr_non_protected_list[2]], bar_width,
            yerr=[std_accuracy_list[2], std_balanced_accuracy_list[2], std_fairness_list[2], std_tpr_protected_list[2], std_tpr_non_protected_list[2], std_tnr_protected_list[2],
                  std_tnr_non_protected_list[2]], label=names[2], color=colors[1],edgecolor='black')

    plt.bar(index + bar_width * 3,
            [accuracy_list[1], balanced_accuracy_list[1], fairness_list[1], tpr_protected_list[1],tpr_non_protected_list[1], tnr_protected_list[1], tnr_non_protected_list[1]], bar_width,
            yerr=[std_accuracy_list[1], std_balanced_accuracy_list[1], std_fairness_list[1], std_tpr_protected_list[1], std_tpr_non_protected_list[1], std_tnr_protected_list[1],
                  std_tnr_non_protected_list[1]], label=names[1], color=colors[3],edgecolor='black')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=4, shadow=False,fancybox=True, framealpha=1.0)
    plt.savefig(output_dir + "_eqop_performance.png",bbox_inches='tight', dpi=200)


def plot_my_results_from_list_dm(dataset):
    names = ['Zafar et al.', 'Krasanakis et al.','Adaboost', 'SMOTEBoost', 'AdaFair']
    output_dir = "Images/DisparateMistreatment/" + dataset

    if dataset == 'adult-gender':
        accuracy_list = [0.84125730476359129, 0.83616398087480067, 0.849160014166814243, 0.8118909107008456,  0.837750132]
        std_accuracy_list = [0.0021313582417819251, 0.0025664235520122067, 0.0017303235479497101, 0.0071879299630220608,  0.0172310617]
        balanced_accuracy_list = [0.70581541065707015, 0.70324222501649845, 0.72621455272353227, 0.8023158733406547, 0.79465430]
        std_balanced_accuracy_list = [0.0034165442777111987, 0.005405353108096869, 0.0017686647761703674,0.02826012688138698, 0.016171962 ]
        fairness_list = [0.131235516797286589, 0.061278466419440502, 0.23200711712535602, 0.4720733687320755933, 0.07795596 ]
        std_fairness_list = [0.010051804326223008, 0.011404127853139124, 0.031476138004416419, 0.034441999118292375, 0.021959274]
        tpr_protected_list = [0.56238015524640588, 0.44872371754247998, 0.362377106310226216, 0.60023359117813, 0.73949702395]
        std_tpr_protected_list = [0.026993921544918097, 0.044443064351168614, 0.040470375545631919,0.025738981830001263, 0.086285605]
        tpr_non_protected_list = [0.44473532321927183, 0.50457550645302358, 0.5357781802276182, 0.8145630805321845,  0.703974812]
        std_tpr_non_protected_list = [0.016463265823206277, 0.051177514224507302, 0.0117798744757660308,0.00827276438672765704,  0.07745906]
        tnr_protected_list = [0.97796080170609046, 0.94624364036822376, 0.994724707529353714, 0.958670520169331951, 0.910980297085 ]
        std_tnr_protected_list = [0.0013381003412159241, 0.039804760376243829, 0.00156165132873236814,0.0012096473014822554,0.04332075278 ]
        tnr_non_protected_list = [0.95283444559215265, 0.973290589834092923, 0.94072027790575674677, 0.710980527646855, 0.8610635112707 ]
        std_tnr_non_protected_list = [0.00151467651774054603, 0.01057896721547606195, 0.0010260785062740344,0.00927085604926498946, 0.05067714210]
    elif dataset == 'compass-gender':
        accuracy_list = [0.6469571807502843, 0.66722508525956803, 0.6672309208033346, 0.65260363774156877, 0.64550966275104]
        std_accuracy_list = [0.007792455261771766, 0.001603059032518792, 0.010757915548752078, 0.012350199687557593, 0.0208 ]
        balanced_accuracy_list = [0.63434455862629, 0.6567458512234059, 0.6600827079741186, 0.6501312277291022, .64886939]
        std_balanced_accuracy_list = [0.01082117593313684, 0.00711351617251441657, 0.0151502593227186553,0.012465191868153471, 0.016222 ]
        fairness_list = [0.0962694551709870803, 0.045651422896865717, 0.36047965671464255, 0.382563401185418984, 0.083480482744 ]
        std_fairness_list = [0.0410244441374238589, 0.0250394653748196136, 0.108798679197871552, 0.103817203788894951, 0.0326794]
        tpr_protected_list = [0.4513301570360016, 0.550033265273026526, 0.42509480150113111, 0.5516417052009246,  0.67064314]
        std_tpr_protected_list = [0.0420115192336582, 0.118528463560619144, 0.08924523659496632, 0.062093212799435625, 0.0757678]
        tpr_non_protected_list = [0.4922756741347097, 0.57121508297727205, 0.63924458215773658, 0.7604757732810334,  0.7120922]
        std_tpr_non_protected_list = [0.04393501872888887, 0.1296676198927925, 0.022886079474549104, 0.02259978132857417, 0.0693820]
        tnr_protected_list = [0.75868457980288374, 0.7695401856952230119, 0.84698719163393386, 0.735589835780717275, 0.5812907277]
        std_tnr_protected_list = [0.03871587937598028, 0.1016389641808471606, 0.0475513476973961929, 0.0524510550598840645,  0.08969849]
        tnr_non_protected_list = [0.7918500480234014, 0.761921714484253183, 0.6995443889322665558, .55006504997761336,  0.5952709007]
        std_tnr_non_protected_list = [0.046849064790145126, 0.101449611884301484, 0.0251608182174372,0.031623538135080789,  0.09202423775]
    elif dataset == 'bank':
        accuracy_list = [0.90009499050094988, 0.899751424857514258, 0.90098790120987901, 0.877931806819318069,  0.8923307669 ]
        std_accuracy_list = [0.0011923676604366244, 0.0019968368708106071, 0.0050694166277808405, 0.006075196953787954, 0.006265161]
        balanced_accuracy_list = [0.65210860705745567, 0.65515041193539315, 0.68265309263554562, 0.7416725434153347, 0.77355314]
        std_balanced_accuracy_list = [0.0049149868796822442, 0.0018187025211562281, 0.01010568728357864683,0.00532987182467402, 0.024169644419]

        fairness_list = [0.030409417988262151, 0.0321728691698796191, 0.1124338491850008172, 0.1145902230433248708, 0.03728204 ]
        std_fairness_list = [0.01831911437453038, 0.010819733007404186, 0.033629513836561771, 0.03714484624130022, 0.02184306287]
        tpr_protected_list = [0.34111771871163785, 0.341121579225188047, 0.3765180304227645969, 0.50867025641738457,  0.6262534]
        std_tpr_protected_list = [0.014701760130666241, 0.0259791705499161388, 0.0216088669221312787,0.0214143152400590156, 0.064571351]
        tpr_non_protected_list = [0.340052713669989998, 0.325166827206530015, 0.46514153412646789, 0.5916543043101094,  0.607502842]
        std_tpr_non_protected_list = [0.0213184925292930135, 0.0562417411490916702, 0.02731227957465693647,0.03133287645748039, 0.063821916]
        tnr_protected_list = [0.97753569792411787, 0.972698784934158359, 0.972594590560276856, 0.9528659813645353,  0.9321964914]
        std_tnr_protected_list = [0.001578348457488543, 0.00929896500655757333, 0.0020801621407839662,0.00114711497242106588, 0.01398628956]
        tnr_non_protected_list = [0.97022120901047203, 0.976247568017577768, 0.948694648329272704, 0.92247093092757714, 0.92008725]
        std_tnr_non_protected_list = [0.0022196508400848152, 0.01005286937213739, 0.00919912565390148657,0.0106487498149501307, 0.0179017613]

    elif dataset == 'kdd':
        accuracy_list = [0.949096930695054227, 0.950397225396443533, 0.94750687970703606,0.9207179754 ]
        std_accuracy_list = [0.01003205607874917051, 0.00120237980150084309, 0.01072928745870637574,  0.01453368869]
        balanced_accuracy_list = [0.5967010511019216046, 0.66139189925026559, 0.7716862451179642632, 0.814263815]
        std_balanced_accuracy_list = [0.0028052829061364568, 0.0066182017362436847, 0.0094180099571768083, 0.024352811]
        fairness_list = [0.021457006162494474, 0.281362278752594369, 0.36327153473224479, 0.0243051 ]
        std_fairness_list = [0.010853469843516568, 0.0103390743780346523, 0.020223839691663668, 0.00955605]
        tpr_protected_list = [0.1847918437742191, 0.14086745275168526, 0.321511600591705491, 0.6906449]
        std_tpr_protected_list = [0.02060130875772270043, 0.00937543168206689258, 0.0014749705252314315,  0.07655150]
        tpr_non_protected_list = [0.200836190539916384, 0.400268244507729271, 0.6423875406492997,0.6932985415 ]
        std_tpr_non_protected_list = [0.0214998153592860202, 0.01036446819123051484, 0.0210881377126951641, 0.066252619]
        tnr_protected_list = [0.9979799508908197, 0.997469878365979992, 0.98779847070972671,  0.9390660470]
        std_tnr_protected_list = [0.00027043660848440748, 0.00021089966275389258, 0.00010913981570608276, 0.0185868434]
        tnr_non_protected_list = [0.99243238104616726, 0.985753894824884225, 0.948535996657650789, 0.93195086]
        std_tnr_non_protected_list = [0.00012111197845660435, 0.0002773489204307783, 0.0020289538389142094,  0.0216364888]

    plt.figure(figsize=(9, 9))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0, 1])
    plt.yticks(numpy.arange(0, 1.01, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 8, step=1.3)
    # index = numpy.arange(7)
    bar_width = 0.175

    plt.xticks(index + 1.5 * bar_width,
               ('Accuracy', 'Bal. Acc.', 'Disp. Mis.', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))

    if dataset != 'kdd':
        colors = ["#ad4ca8", "#3498DB", "#B0B0B0","#E74C3C", "#2ECC71"]
        names = ['Zafar et al.', 'Krasanakis et al.','Adaboost', 'SMOTEBoost', 'AdaFair']
    else:
        colors = [ "#3498DB", "#B0B0B0", "#E74C3C", "#2ECC71"]
        names = ['Krasanakis et al.', 'Adaboost', 'SMOTEBoost', 'AdaFair']

    for i in range(0, len(names)):

        plt.bar(index + bar_width * i,
                [accuracy_list[i], balanced_accuracy_list[i], fairness_list[i], tpr_protected_list[i],
                 tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
                yerr=[std_accuracy_list[i], std_balanced_accuracy_list[i], std_fairness_list[i], std_tpr_protected_list[i],
                      std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]], label=names[i], color=colors[i], edgecolor='black')

    if dataset != 'kdd':
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.106), ncol=3, shadow=False, fancybox=True, framealpha=1.0)
    else:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.106), ncol=2, shadow=False, fancybox=True, framealpha=1.0)
    plt.savefig(output_dir + "_dm_performance.png", bbox_inches='tight', dpi=200)


#


#plot_my_results_from_list_sp('adult-gender')
#plot_my_results_from_list_sp('bank')
#plot_my_results_from_list_sp('compass-gender')
#plot_my_results_from_list_sp('kdd')

# plot_my_results_from_list_eqop('adult-gender')
# plot_my_results_from_list_eqop('bank')
# plot_my_results_from_list_eqop('compass-gender')
# plot_my_results_from_list_eqop('kdd')

# plot_my_results_from_list_dm('adult-gender')
# plot_my_results_from_list_dm('bank')
# plot_my_results_from_list_dm('compass-gender')
# plot_my_results_from_list_dm('kdd')

def plot_my_results_single_vs_amort_eqop(results, names, output_dir, dataset):
    accuracy_list = []
    balanced_accuracy_list = []
    fairness_list = []
    tpr_protected_list = []
    tpr_non_protected_list = []
    tnr_protected_list = []
    tnr_non_protected_list = []
    std_accuracy_list = []
    std_balanced_accuracy_list = []
    std_fairness_list = []
    std_tpr_protected_list = []
    std_tpr_non_protected_list = []
    std_tnr_protected_list = []
    std_tnr_non_protected_list = []

    for list_of_results in results:

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in list_of_results:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["TPR_protected"])
            tpr_non_protected.append(item["TPR_non_protected"])
            tnr_protected.append(item["TNR_protected"])
            tnr_non_protected.append(item["TNR_non_protected"])

        numpy.mean(accuracy)
        numpy.std(accuracy)

        accuracy_list.append(numpy.mean(accuracy))
        balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        fairness_list.append(numpy.mean(fairness))
        tpr_protected_list.append(numpy.mean(tpr_protected))
        tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        tnr_protected_list.append(numpy.mean(tnr_protected))
        tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

        std_accuracy_list.append(numpy.std(accuracy))
        std_balanced_accuracy_list.append(numpy.std(balanced_accuracy))
        std_fairness_list.append(numpy.std(fairness))
        std_tpr_protected_list.append(numpy.std(tpr_protected))
        std_tpr_non_protected_list.append(numpy.std(tpr_non_protected))
        std_tnr_protected_list.append(numpy.std(tnr_protected))
        std_tnr_non_protected_list.append(numpy.std(tnr_non_protected))

    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1.01, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 5, step=1)
    # index = numpy.arange(7)
    bar_width = 0.25

    plt.xticks(index + 1.5*bar_width ,
               ('Eq. Op.', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))

    colors = ['#34495E','#2ECC71','r','c','m','y','k', 'dimgray']
    for i in range(0, len(names)):
        plt.bar(index + bar_width * i,
                [ fairness_list[i], tpr_protected_list[i],
                 tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
                yerr=[std_fairness_list[i],
                      std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]],
                label=names[i], color=colors[i],edgecolor='black')

    plt.legend(loc='best',ncol=2,bbox_to_anchor=(1.05, 1.105),   shadow=False,fancybox=True, framealpha=1.0)
    # plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir + ".png",bbox_inches='tight', dpi=400)



def plot_my_results_single_vs_amort_dm(results, names, output_dir, dataset):
    accuracy_list = []
    balanced_accuracy_list = []
    fairness_list = []
    tpr_protected_list = []
    tpr_non_protected_list = []
    tnr_protected_list = []
    tnr_non_protected_list = []
    std_accuracy_list = []
    std_balanced_accuracy_list = []
    std_fairness_list = []
    std_tpr_protected_list = []
    std_tpr_non_protected_list = []
    std_tnr_protected_list = []
    std_tnr_non_protected_list = []

    for list_of_results in results:

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in list_of_results:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["TPR_protected"])
            tpr_non_protected.append(item["TPR_non_protected"])
            tnr_protected.append(item["TNR_protected"])
            tnr_non_protected.append(item["TNR_non_protected"])

        numpy.mean(accuracy)
        numpy.std(accuracy)

        accuracy_list.append(numpy.mean(accuracy))
        balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        fairness_list.append(numpy.mean(fairness))
        tpr_protected_list.append(numpy.mean(tpr_protected))
        tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        tnr_protected_list.append(numpy.mean(tnr_protected))
        tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

        std_accuracy_list.append(numpy.std(accuracy))
        std_balanced_accuracy_list.append(numpy.std(balanced_accuracy))
        std_fairness_list.append(numpy.std(fairness))
        std_tpr_protected_list.append(numpy.std(tpr_protected))
        std_tpr_non_protected_list.append(numpy.std(tpr_non_protected))
        std_tnr_protected_list.append(numpy.std(tnr_protected))
        std_tnr_non_protected_list.append(numpy.std(tnr_non_protected))

    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1.01, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 5, step=1)
    # index = numpy.arange(7)
    bar_width = 0.25

    plt.xticks(index + 1.5*bar_width ,
               ('Disp. Mis.', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))

    colors = ['#34495E','#2ECC71','r','c','m','y','k', 'dimgray']
    for i in range(0, len(names)):
        plt.bar(index + bar_width * i,
                [ fairness_list[i], tpr_protected_list[i],
                 tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
                yerr=[std_fairness_list[i],
                      std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]],
                label=names[i], color=colors[i],edgecolor='black')

    plt.legend(loc='best',ncol=2,bbox_to_anchor=(1.05, 1.105),   shadow=False,fancybox=True, framealpha=1.0)
    # plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir + ".png",bbox_inches='tight', dpi=400)

def plot_my_results_single_vs_amort_sp(results, names, output_dir, dataset):
    accuracy_list = []
    balanced_accuracy_list = []
    fairness_list = []
    tpr_protected_list = []
    tpr_non_protected_list = []
    tnr_protected_list = []
    tnr_non_protected_list = []
    std_accuracy_list = []
    std_balanced_accuracy_list = []
    std_fairness_list = []
    std_tpr_protected_list = []
    std_tpr_non_protected_list = []
    std_tnr_protected_list = []
    std_tnr_non_protected_list = []

    for list_of_results in results:

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in list_of_results:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["Positive_prot_pred"])
            tpr_non_protected.append(item["Positive_non_prot_pred"])
            tnr_protected.append(item["Negative_prot_pred"])
            tnr_non_protected.append(item["Negative_non_prot_pred"])

        numpy.mean(accuracy)
        numpy.std(accuracy)

        accuracy_list.append(numpy.mean(accuracy))
        balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        fairness_list.append(numpy.mean(fairness))
        tpr_protected_list.append(numpy.mean(tpr_protected))
        tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        tnr_protected_list.append(numpy.mean(tnr_protected))
        tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

        std_accuracy_list.append(numpy.std(accuracy))
        std_balanced_accuracy_list.append(numpy.std(balanced_accuracy))
        std_fairness_list.append(numpy.std(fairness))
        std_tpr_protected_list.append(numpy.std(tpr_protected))
        std_tpr_non_protected_list.append(numpy.std(tpr_non_protected))
        std_tnr_protected_list.append(numpy.std(tnr_protected))
        std_tnr_non_protected_list.append(numpy.std(tnr_non_protected))

    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1.01, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 5, step=1)
    # index = numpy.arange(7)
    bar_width = 0.25

    # plt.xticks(index + 1.5*bar_width ,('Eq. Op.', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))
    plt.xticks(index + 1.5*bar_width , ('St.Parity', 'Prot. Pos.', 'Non-Prot. Pos.', 'Prot. Neg.', 'Non-Prot. Neg.'))

    colors = ['#34495E','#2ECC71','r','c','m','y','k', 'dimgray']
    for i in range(0, len(names)):
        plt.bar(index + bar_width * i,
                [ fairness_list[i], tpr_protected_list[i],
                 tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
                yerr=[std_fairness_list[i],
                      std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]],
                label=names[i], color=colors[i],edgecolor='black')

    plt.legend(loc='best',ncol=2,bbox_to_anchor=(1.05, 1.105),   shadow=False,fancybox=True, framealpha=1.0)
    # plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir + ".png",bbox_inches='tight', dpi=400)
