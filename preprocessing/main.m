@import Foundation;
@import AppKit;

#import "correct.h"

int main(int argc, char *argv[]) {
    @autoreleasepool {
        Corrector *c = [[Corrector alloc] initWithDataset:@"../../dataset/train_neg_full.txt" output:@"../../dataset/train_neg_full_corrected.txt"];
        [c test];
    }
}