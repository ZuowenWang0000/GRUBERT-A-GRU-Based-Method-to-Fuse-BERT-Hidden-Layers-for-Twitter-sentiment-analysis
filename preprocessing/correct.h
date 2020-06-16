#import <stdlib.h>

@interface Corrector : NSObject {
    FILE *fp;
    FILE *out;
    NSSpellChecker *checker;
}
@property (strong, nonatomic) NSDictionary *emoticons;
- (instancetype)initWithDataset:(NSString *)dataset output:(NSString *)output_path;
- (void)test;
@end