from torchmetrics import WordErrorRate as WER

expected = "This is a test sentence"
target = "This is a different test sentence"

def main():
    metric = WER()
    print(metric(expected, target))

main()
