import settings


class Benchmark:
    def __init__(self, actual=None, predicted=None):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.p = 0
        self.n = 0

        self.dt = []

        if actual is not None and predicted is not None:
            self.calculateConfusionMatrix(actual, predicted)

    def __iadd__(self, other):
        if type(other) is Benchmark:
            self.tp += other.tp
            self.fp += other.fp
            self.tn += other.tn
            self.fn += other.fn
            self.p += other.p
            self.n += other.n
            self.dt += other.dt
        return self

    def setConfusionMatrix(self, tp, fp, tn, fn):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

        self.p = self.tp + self.fn
        self.n = self.tn + self.fp

    def calculateConfusionMatrix(self, predicted, actual):
        """
        Takes actual and predicted values and stores the confusion matrix by
        looping trough the actual and predicted values, converting them to 1
        and 0, and comapring them.
        """
        if settings.singleTargetValue:
            predicted = [predicted]
            actual = [actual]
        
        currentLeak = False
        for i, (x, y) in enumerate(zip(predicted, actual)):
            x = int(x)
            y = int(y)

            if y == 0:
                if x == 0:
                    self.tn += 1
                else:
                    self.fp += 1

                currentLeak = False
            elif y == 1:
                if x == 1:
                    self.tp += 1
                else:
                    self.fn += 1

                # Calculate detection time, if the leak isnt detected it is
                # shown as -1
                if not currentLeak:
                    for j, x in enumerate(predicted[i:]):
                        if int(x) == 1:
                            self.dt.append(j)
                            break
                        elif j == len(predicted[i:])-1:
                            self.dt.append(-1)
                currentLeak = True

            else:
                raise RuntimeError(
                    "Actual value is not comaprable to either 1 or 0")
        self.p = self.tp + self.fn
        self.n = self.tn + self.fp

    def getTPR(self):
        if self.p == 0:
            return 1.0
        else:
            return self.tp/self.p

    def getFPR(self):
        if self.n == 0:
            return 1.0
        else:
            return self.fp/self.n

    def getAccuracy(self):
        return (self.tp+self.tn)/(self.p+self.n)

    def getDT(self):
        return self.dt


if __name__ == "__main__":
    b = Benchmark()
    x = [0, 0, 0, 0, 0]
    y = [0, 1, 0, 0, 0]
    b.calculateConfusionMatrix(x, y)
    print(b.getDT())
