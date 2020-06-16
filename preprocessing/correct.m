@import Foundation;
@import AppKit;

#import <assert.h>
#import "correct.h"

@implementation Corrector
- (instancetype)initWithDataset:(NSString *)dataset output:(NSString *)output_path {
    self = [super init];
    if (self) {
        fp = fopen([dataset UTF8String], "r");
        out = fopen([output_path UTF8String], "w");
        checker = [NSSpellChecker sharedSpellChecker];
    }
    self.emoticons = @{
        @"=|": @"xxemotneutral",
        @"=-(": @"xxemotfrown",
        @"=-)": @"xxemotsmile",
        @"=:": @"xxemotneutral",
        @"=/": @"xxemotfrown",
        @"='(": @"xxemotfrown",
        @"='[": @"xxemotfrown",
        @"=(": @"xxemotfrown",
        @"=)": @"xxemotsmile",
        @"=[": @"xxemotfrown",
        @"=]": @"xxemotsmile",
        @"={": @"xxemotfrown",
        @"=\\": @"xxemotfrown",
        @">=(": @"xxemotfrown",
        @">=)": @"xxemotsmile",
        @">:|": @"xxemotneutral",
        @">:/": @"xxemotfrown",
        @">:[": @"xxemotfrown",
        @">:@": @"xxemotfrown",
        @"|:": @"xxemotneutral",
        @";|": @"xxemotneutral",
        @";-}": @"xxemotsmile",
        @";:": @"xxemotneutral",
        @";/": @"xxemotfrown",
        @";'/": @"xxemotfrown",
        @";'(": @"xxemotfrown",
        @";')": @"xxemotsmile",
        @";)": @"xxemotsmile",
        @";]": @"xxemotsmile",
        @";}": @"xxemotsmile",
        @";*{": @"xxemotfrown",
        @":|": @"xxemotneutral",
        @":-|": @"xxemotneutral",
        @":-/": @"xxemotfrown",
        @":-[": @"xxemotfrown",
        @":-]": @"xxemotsmile",
        @":-}": @"xxemotsmile",
        @":-@": @"xxemotneutral",
        @":-\\": @"xxemotfrown",
        @":;": @"xxemotneutral",
        @"::": @"xxemotneutral",
        @":/": @"xxemotfrown",
        @":'|": @"xxemotneutral",
        @":'/": @"xxemotfrown",
        @":')": @"xxemotsmile",
        @":'{": @"xxemotfrown",
        @":'}": @"xxemotsmile",
        @":'\\": @"xxemotneutral",
        @":(": @"xxemotfrown",
        @":)": @"xxemotsmile",
        @":]": @"xxemotsmile",
        @":[": @"xxemotfrown",
        @":{": @"xxemotfrown",
        @":}": @"xxemotsmile",
        @":@": @"xxemotneutral",
        @":*{": @"xxemotfrown",
        @":\\": @"xxemotfrown",
        @"(=": @"xxemotsmile",
        @"(;": @"xxemotsmile",
        @"(':": @"xxemotsmile",
        @")=": @"xxemotfrown",
        @")':": @"xxemotfrown",
        @"[;": @"xxemotsmile",
        @"]:": @"xxemotfrown",
        @"{:": @"xxemotsmile",
        @"\\=": @"xxemotfrown",
        @"\\:": @"xxemotfrown"
    };
    return self;
}

- (void)test {
    NSMutableSet<NSMutableString *> *set = [[NSMutableSet alloc] initWithCapacity:1250000];
    NSArray<NSString *> *emots = [self.emoticons allKeys];
    // NSMutableArray<NSString *> *temp = [[NSMutableArray alloc] init];
    // for (NSString *emot in emots) {
    //     [temp addObject:[NSRegularExpression escapedPatternForString:emot]];
    // }
    // NSString *regex = [temp componentsJoinedByString:@"|"];
    // NSRegularExpression emoticons_regex = [NSRegularExpression expressionWithPattern:regex];
    // NSLog(@"%@", regex);
    char buf[281];
    NSString *prev = @"";
    int limit = 1000;
    int i = 0;
    while (fgets(buf, 280, fp) != NULL && i++ < limit) {
        NSMutableString *str = [NSMutableString stringWithUTF8String:buf];
        if ([str compare:prev] == NSOrderedSame) {
            continue;
        }
        prev = [NSString stringWithString:str];
        [str replaceOccurrencesOfString:@"<user>" withString:@"xxuser" options:NSLiteralSearch range:NSMakeRange(0, [str length])];
        [str replaceOccurrencesOfString:@"<url>" withString:@"xxurl" options:NSLiteralSearch range:NSMakeRange(0, [str length])];
        [str replaceOccurrencesOfString:@"#" withString:@"# " options:NSLiteralSearch range:NSMakeRange(0, [str length])];
        [str replaceOccurrencesOfString:@"\\s+" withString:@" " options:NSLiteralSearch range:NSMakeRange(0, [str length])];
        for (NSString *emot in emots) {
            [str replaceOccurrencesOfString:emot withString:self.emoticons[emot] options:NSLiteralSearch range:NSMakeRange(0, [str length])];
        }
        NSInteger wordCount;
        NSRange range = NSMakeRange(0, 0);
        while (range.location != NSNotFound) {
            range = [checker checkSpellingOfString:str startingAt:(range.location+range.length) language:[checker language] wrap:NO inSpellDocumentWithTag:0 wordCount:&wordCount];
            // printf("checked spelling\n");
            // NSLog(@"%@", NSStringFromRange(range));
            if (range.location != NSNotFound) {
                // NSLog(@"%ld", wordCount);
                // NSLog(@"%@", NSStringFromRange(range));
                // NSLog(@"%@", [str substringWithRange:range]);
                // NSLog(@"%@", [checker correctionForWordRange:range inString:str language:[checker language] inSpellDocumentWithTag:0]);
                NSArray<NSString *> *guesses = [checker guessesForWordRange:range inString:str language:[checker language] inSpellDocumentWithTag:0];
                // printf("got guesses\n");
                // NSLog(@"%@", guesses);
                if ([guesses count] > 0) {
                    [str replaceCharactersInRange:range withString:[guesses objectAtIndex:0]];
                }
            } else {
                break;
            }
        }
        [str replaceOccurrencesOfString:@"rt." withString:@"rt" options:NSLiteralSearch range:NSMakeRange(0, [str length])];

        // NSLog(@"final string: %@", str);
        // fputs([str UTF8String], out);
        [set addObject:str];
    }
    for (NSMutableString *string in set) {
        fputs([[string lowercaseString] UTF8String], out);
    }
}
@end