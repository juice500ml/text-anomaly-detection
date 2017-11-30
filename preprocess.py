from konlpy.tag import Twitter
twitter = Twitter()

def iskor(c):
    return c>='가' and c<='힣'
def isalnum(c):
    return (c>='a' and c<='z') or (c>='A' and c<='Z') or (c>='0' and c<='9')

def diffchar(c1, c2):
    if c1.isspace() or c2.isspace(): return False
    if isalnum(c1) and isalnum(c2): return False
    if iskor(c1) and iskor(c2): return False
    return True

eng = [l.strip() for l in open('dict.txt') if len(l) > 2]
cnt = 0
with open('onlyprod_twitter_total.txt', 'w') as f:
    for line in open('./orig/product.txt'):
        l = list(line.split('\x01')[2])

        # 특수기호 제거
        l = [c if c.isalnum() or c.isspace() else ' ' for c in l]
        
        # 서로 다른 언어로 되어있으면 분리
        idx = 1
        while idx < len(l):
            if diffchar(l[idx-1], l[idx]):
                l.insert(idx, ' ')
            idx += 1

        l = (''.join(l)).split()
        
        # 소문자로 바꾸기
        l = [c.lower() for c in l]
        # 사전에 있는 영단어만 남기기
        l = [c for c in l if (iskor(c[0]) or (c in eng))]

        # konlpy
        idx = 0
        while idx < len(l):
            if iskor(l[idx][0]):
                sp = twitter.morphs(l[idx])
                if idx == 0: l = sp + l[idx+1:]
                else: l = l[:idx] + sp + l[idx+1:]
                idx += len(sp)
            else:
                idx += 1
        # konlpy

        result = ' '.join(l) + '\n'
        # preprocess failed
        if len(result) == 1:
            result = line.split('\x01')[2]
        f.write(result)

        cnt += 1
        if cnt % 100000 == 0:
            print(cnt)
