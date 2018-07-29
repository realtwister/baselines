import scipy.signal;

def discount(x):
    return scipy.signal.lfilter([1], [1, -0.99], x[::-1], axis=0)[::-1];

a = [1,1,1];

print(a);
print(discount(a));
