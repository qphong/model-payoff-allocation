import operator as op
from functools import reduce
import itertools
import math
import numpy as np
from numpy.lib.function_base import disp


def nCr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom  # or / in Python 2


def get_conditional_shapley(n, v, i):
    """Compute the conditional Shapley values

    Requires:
        n: the number of parties in the grand coalition
        v: the characteristic value function (the argument is bitmask)
            v(0): value of empty coalition
            v(1 << n - 1): value of the grand coalition
    """
    contrib_j2i = [0.0] * n

    for c in range(1 << (n - 2)):
        c_debug = c 

        if i < n - 2:
            # swap bit i and bit n-2
            c = (((1 << i) & c) << (n - 2 - i)) | c
            # clear bit i (since bit n-2 is always 0 when we consider only n-2 bits)
            # but we do not need to clear bit i, because we always set bit i on the next step
        # turn on bit i
        c = (1 << i) | c

        for j in range(n):
            if j == i:
                continue

            if j < n - 1:
                if i < n-1:
                    # swap bit j and bit n-1
                    c_i = (((1 << j) & c) << (n - 1 - j)) | c
                    # clear bit j (since bit n-1 is always 0 when we consider only n-2 bits and i < n-1)
                    c_i = (~(1 << j)) & c_i
                else:
                    # swap bit j and bit n-2
                    c_i = (((1 << j) & c) << (n - 2 - j)) | c
                    # clear bit j (since bit n-2 is always 0 when we consider only n-2 bits and i == n-1)
                    c_i = (~(1 << j)) & c_i
            else:
                c_i = c

            # turn on bit j
            c_ij = (1 << j) | c_i

            # c_ij is subset containing both i and j
            # c_i is c_ij \ {j}

            marginal_contrib_j = v(c_ij) - v(c_i)

            # print("==")
            # print(bin(c_ij), bin(c_i), marginal_contrib_j)

            # print("\nCoalitions:", bin(c_debug), bin(c), bin(c_ij), bin(c_i))

            # count bit set in c_j
            n_bit_set = 0
            # number of parties excluding j
            #    still including i
            while c_i:
                c_i &= c_i - 1
                n_bit_set += 1

            contrib_j2i[j] += (
                1.0 / (n - 1.0) / nCr(n - 2, n_bit_set - 1) * marginal_contrib_j
            )

            # print("Contribution of", j, "to", i, ":", marginal_contrib_j)
            # print("Normalize:", 1 / (n-1.0) / nCr(n-2, n_bit_set-1))

    return contrib_j2i


def get_bitmask4coalition(coalition):
    # coalition is a list of parties, e.g., [0,1,3]
    bitmask = 0
    for i in coalition:
        bitmask |= 1 << i
    return bitmask


def get_payoff_flow_unlimited_budget(n, v):
    # conditional_shapley_vals: list of all conditional shapley values
    # return 2d ndarray a
    # a[i,j]: flow of payoff from i to j
    payoff_flow = []
    for i in range(n):
        i2others = get_conditional_shapley(n, v, i)
        payoff_flow.append(i2others)

    return np.array(payoff_flow)


def get_payoff_flow(n, v, budget=None, epsilon=1e-12):
    payoff_flow = get_payoff_flow_unlimited_budget(n, v)  # no budget constraint
    outgoing_payoff = np.sum(payoff_flow, axis=1)  # outgoing payoff
    incoming_payoff = np.sum(payoff_flow, axis=0)  # incoming payoff
    income = incoming_payoff - outgoing_payoff
    model_reward = outgoing_payoff
    for i in range(n):
        model_reward[i] += v(1 << i)

    if budget is None:
        return payoff_flow, income, model_reward

    # if income is negative and income + budget < 0, then the budget is exceeded
    # income >= - outgoing_payoff because outgoing_payoff >= 0 and incoming_payoff >= 0

    while np.any(income + budget < -epsilon):
        exceed_idxs = np.where(income + budget < -epsilon)[0]
        i = exceed_idxs[0]
        reduction = income[i] + budget[i]  # negative


        # this reduction only affects conditional shapley given i
        # every coalition in the permutations given i
        # is reduced by a fraction reduction / outgoing_payoff[i]
        # due to linearity property of conditional Shapley value
        # update payoff_flow
        payoff_flow[i, :] = payoff_flow[i, :] * (1 + reduction / outgoing_payoff[i])

        print("allocation.py:get_payoff_flow Reduction: ", reduction)
        print("  party {} weight: {}".format(i, (1 + reduction / outgoing_payoff[i])))
        outgoing_payoff = np.sum(payoff_flow, axis=1)  # outgoing payoff
        incoming_payoff = np.sum(payoff_flow, axis=0)  # incoming payoff
        income = incoming_payoff - outgoing_payoff

    payoff_flow[np.where(np.abs(payoff_flow) < epsilon)] = 0
    income[np.where(np.abs(income) < epsilon)] = 0
    
    model_reward = outgoing_payoff
    for i in range(n):
        model_reward[i] += v(1 << i)

    return payoff_flow, income, model_reward


def test_v2(c):
    # n = 5
    # count bit set in c
    if c == 0:
        return 0

    n_bit_set = 0
    while c:
        c &= c - 1
        n_bit_set += 1
    
    return 2**n_bit_set


def test_v(c):
    # n = 3
    """
    v(0) = 1
    v(1) = 1
    v(2) = 3
    v(01) = 2
    v(02) = 3
    v(12) = 3
    v(012) = 3

    v(i) = 1 if i = 0, 1
    v(i) = 3 if i = 2, 02, 12, 012
    v(i) = 2 if i = 01

    given 1:
        102:
            phi(0) = 0.5[v(01) - v(1)] = 0.5
            phi(2) = 0.5[v(012) - v(01)] = 0.5
        120
            phi(0) = 0
            phi(2) = 0.5[v(12) - v(1)] = 1
        Hence,
            phi(0) = 0.5,
            phi(1) = 1.5
    """
    if c == 0:
        return 0
    elif c in map(get_bitmask4coalition, [[0], [1]]):
        return 1.0
    elif c == get_bitmask4coalition([0, 1]):
        return 2.0
    else:
        return 3.
    # elif c in map(get_bitmask4coalition, [[1, 2], [0, 1, 2]]):
    #     return 4.0
    # else:
    #     return 3.0


def test_v1(c):
    # n = 3
    """
    v(i) = 1 if i = 0, 1
    v(i) = 3 if i = 2, 02, 12, 012
    v(i) = 2 if i = 01

    given 1:
        102:
            phi(0) = 0.5[v(01) - v(1)] = 0.5
            phi(2) = 0.5[v(012) - v(01)] = 0.5
        120
            phi(0) = 0
            phi(2) = 0.5[v(12) - v(1)] = 1
        Hence,
            phi(0) = 0.5,
            phi(1) = 1.5
    """
    if c == 0:
        return 0
    elif c in map(get_bitmask4coalition, [[0], [1]]):
        return 1.0
    elif c == get_bitmask4coalition([0, 1]):
        return 2.0
    else:
        return 3.0


def test_conditional_shapley():
    n = 3

    contrib_jto0 = get_conditional_shapley(n, test_v, 0)
    assert contrib_jto0[0] == 0.0
    assert contrib_jto0[1] == 0.5
    assert contrib_jto0[2] == 1.5

    contrib_jto1 = get_conditional_shapley(n, test_v, 1)
    assert contrib_jto1[0] == 0.5
    assert contrib_jto1[1] == 0.0
    assert contrib_jto1[2] == 1.5

    contrib_jto2 = get_conditional_shapley(n, test_v, 2)
    assert contrib_jto2[0] == 0.0
    assert contrib_jto2[1] == 0.0
    assert contrib_jto2[2] == 0.0


def get_payoff_flow_unlimited_budget_with_payment_distribution(n, v, pay_dist):
    """This function is to test if the payment scheme is independent of the distribution of the payment to previous parties!

    n: number of parties in the grand coalition
    v: the characteristic value function (the argument is bitmask)
    pay_dist: the distribution over P_pi^i indicates the proportion each party in
        P_pi^i needs to pay to party i due to its contribution v(P_pi^i cup i)  - v(P_pi^i)
        pay_dist: list of ndarray
            pay_dist[0]: empty array
            pay_dist[i]: ndarray of size i (distribution to the previous i parties)
    """
    payoff = np.zeros([n, n])

    for perm in itertools.permutations(list(range(n))):
        for i, party in enumerate(perm):
            if i > 0:
                coalition_bef_i = perm[:i]
                bitmask_coalition_bef_i = get_bitmask4coalition(coalition_bef_i)
                bitmask_coalition_incl_i = bitmask_coalition_bef_i | (1 << party)
                marginal_contrib = v(bitmask_coalition_incl_i) - v(
                    bitmask_coalition_bef_i
                )
                payments = pay_dist[i] * marginal_contrib

                for j, partyj in enumerate(perm[:i]):
                    payoff[partyj, party] += payments[j]

    factorial_n = math.factorial(n)
    payoff /= factorial_n

    return payoff


def empirically_prove_distribution_independent_payment():

    n = 7

    # randomly constructing v
    v_memoiz = np.zeros(1 << n)
    for i in range(1 << n):
        v_memoiz[i] = np.random.rand() * 10.0

    v = lambda c: v_memoiz[c]

    uniform_pay_dist = [None] * n
    for i in range(n):
        uniform_pay_dist[i] = np.ones(i)
        if i > 0:
            uniform_pay_dist[i] /= i

    first_pay_dist = [None] * n
    for i in range(n):
        first_pay_dist[i] = np.zeros(i)
        if i > 0:
            first_pay_dist[i][0] = 1.0

    last_pay_dist = [None] * n
    for i in range(n):
        last_pay_dist[i] = np.zeros(i)
        if i > 0:
            last_pay_dist[i][-1] = 1.0

    rand_pay_dist = [None] * n
    for i in range(n):
        rand_pay_dist[i] = np.random.rand(i)
        if i > 0:
            rand_pay_dist[i] /= np.sum(rand_pay_dist[i])

    print("payment distributions:")
    for payment_dists in [
        uniform_pay_dist,
        first_pay_dist,
        last_pay_dist,
        rand_pay_dist,
    ]:
        print("vvvvvvvvv")
        for i in payment_dists:
            print(i)
        print("=========")

    uniform_payoff = get_payoff_flow_unlimited_budget_with_payment_distribution(
        n, v, uniform_pay_dist
    )
    first_payoff = get_payoff_flow_unlimited_budget_with_payment_distribution(
        n, v, first_pay_dist
    )
    last_payoff = get_payoff_flow_unlimited_budget_with_payment_distribution(
        n, v, last_pay_dist
    )
    rand_payoff = get_payoff_flow_unlimited_budget_with_payment_distribution(
        n, v, rand_pay_dist
    )

    print(uniform_payoff)
    print(np.sum(np.abs(uniform_payoff - first_payoff)))
    print(np.sum(np.abs(first_payoff - last_payoff)))
    print(np.sum(np.abs(last_payoff - rand_payoff)))


"""
prove that the allocation does not depend on the order of choosing violated party
    prove that the allocation results in maximum incoming (and outcoming) payoff
        sum(payoff) maximized given the budget
        suppose sum(payoff) < sum(payoff*)
        then 
            exist i,j
                payoff[i,j] < payoff*[i,j]
                => payoff[i,:] < payoff*[i,:]
                this happens when sum(payoff[:,i]) < sum(payoff*[:,i]) (if not, the algorithm shouldn't decrease payoff[i,:] that much) (*)
                => exist k != i, such that payoff[k,i] < payoff*[k,i]
                => payoff[k,:] < payoff*[k,:]

                (*) consider the first time the update makes sum(payoff[:,i]) < sum(payoff*[:,i])
                    suppose this update is for party k != i
                        payoff[k,i] < payoff*[k,i]

        suppose sum(payoff) < sum(payoff*)
        consider the first time, that update makes this happens
            suppose it is at party i
                payoff[i,:] < payoff*[i,:]
                but this requires the sum(payoff[:,i]) < sum(payoff*[:,i])
                this cannot happen for positive payoff[i,j] (because before this time, sum(payoff[:,i]) decrease)

        sum(payoff) decrease at each iteration
        in particular only 1 i, sum(payoff[i,:]) decrease
        consider the first time sum(payoff) < sum(payoff*)
            it is because sum(payoff[i,:]) < sum(payoff*[i,:])
                happens because sum(payoff[:,i]) < sum(payoff*[:,i]) but this cannot happen either because we always reduce the incoming of each party in each iteration

        the incoming_payoff is maximized
        the outgoing_payoff is maximized

        at every iteration, 
            total income i reduces
            total expense i reduces 
        
        need to consider the constraint


    prove the uniqueness
        if not unique:
            sum(payoff1) = sum(payoff2)
            but payoff1[i,j] < payoff2[i,j]
                payoff1[i,:] < payoff2[i,:]
                and
                payoff1[j,:] > payoff2[j,:]

                =>
                payoff1[:,i] < payoff2[:,i]

                some k payoff1[k,i] < payoff2[k,i]

                suppose the first time 
                    payoff1[i,:] < payoff2[i,:]

                the expense of each is maximum that satisfies the constraint
                    for example, one expense is not
                        consider the first time 
                            => another expense less than the maximum contradition!

                some expend more, then other expend more, therefore, we can define the notion of maximum 
                    (a,b) (c,d)
                    a > c, b < d?
                    a > c and both are within budget 
                    incoming a expend b

                    maximum expense with respect to the budget: a vector that dominates all other valid expense vector
                    suppose there is a valid vector payoff* that is not dominated by the result payoff
                        there exists i, 
                            sumpayoff[i,:] < sumpayoff*[i,:]
                            consider the first time sumpayoff[i,:] < sumpayoff*[i,:] -> contradition
                    => the result payoff dominates any other valid vector 
                    => the maximum possible expense
                        if budget + income < 0:
                            reduction = budget + income in the performance 
                        else
                            reduction = 0 reduction in the performance is 0
                            

                fixed payoff2, consider the first time
                    payoff1[i,:] < payoff2[i,:]

                implies
                sum(payoff1[:,i]) < sum(payoff2[:,i])
                => payoff1[k,i] < payoff2[k,i]
                => sum(payoff)

"""
