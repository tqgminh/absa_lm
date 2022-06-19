mkdir weight
mkdir weight/res14
mkdir weight/res16

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-j2lxcITUSIaBgKiYfmEuK_n_BORjcJN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-j2lxcITUSIaBgKiYfmEuK_n_BORjcJN" -O bert-res14.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=11i2AFwb27xIGGeeGyPueCHRh3E0_mgb6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11i2AFwb27xIGGeeGyPueCHRh3E0_mgb6" -O meta-res14-blending-linear.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-AiDE4dUg9Ocr6aTDHrKdzHhgCE8W0Md' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-AiDE4dUg9Ocr6aTDHrKdzHhgCE8W0Md" -O meta-res14-blending-mlp.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-Yb5O6SnmNxxqUEJXaqGJo_gmKWBeqqH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-Yb5O6SnmNxxqUEJXaqGJo_gmKWBeqqH" -O roberta-res14.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14uKsDxWof5Z4fliNhpf34a76aQKnSlHU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14uKsDxWof5Z4fliNhpf34a76aQKnSlHU" -O xlmr-res14.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wbe9mksL7-fFqmdFyW0TOKVA7PUJbv5D' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wbe9mksL7-fFqmdFyW0TOKVA7PUJbv5D" -O xlnet-res14.pth && rm -rf /tmp/cookies.txt

mv bert-res14.pth weight/res14
mv meta-res14-blending-linear.pth weight/res14
mv meta-res14-blending-mlp.pth weight/res14
mv roberta-res14.pth weight/res14
mv xlmr-res14.pth weight/res14
mv xlnet-res14.pth weight/res14

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-ubfhtLw6NALGSxzbrMO74GdaeXZlO4Z' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-ubfhtLw6NALGSxzbrMO74GdaeXZlO4Z" -O bert-res16.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10Zzyq28U60Zunck2-KzVGwFGwAHsXcEB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10Zzyq28U60Zunck2-KzVGwFGwAHsXcEB" -O meta-res16-blending-linear.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-QV_r3Aygft2BlX75LB3MOuNk9parghy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-QV_r3Aygft2BlX75LB3MOuNk9parghy" -O meta-res16-blending-mlp.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-gSriHGICjnT7gETcfa7xtfwDqMep_mz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-gSriHGICjnT7gETcfa7xtfwDqMep_mz" -O roberta-res16.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-BruVhjr5R6HG7hEt_gbEYmL5mCcUmB1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-BruVhjr5R6HG7hEt_gbEYmL5mCcUmB1" -O xlmr-res16.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-CF5VzaIPBsdO-je9Vz6EJhV_EgXtdcm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-CF5VzaIPBsdO-je9Vz6EJhV_EgXtdcm" -O xlnet-res16.pth && rm -rf /tmp/cookies.txt

mv bert-res16.pth weight/res16
mv meta-res16-blending-linear.pth weight/res16
mv meta-res16-blending-mlp.pth weight/res16
mv roberta-res16.pth weight/res16
mv xlmr-res16.pth weight/res16
mv xlnet-res16.pth weight/res16

# https://drive.google.com/file/d/1-ubfhtLw6NALGSxzbrMO74GdaeXZlO4Z/view?usp=sharing
# https://drive.google.com/file/d/10Zzyq28U60Zunck2-KzVGwFGwAHsXcEB/view?usp=sharing
# https://drive.google.com/file/d/1-QV_r3Aygft2BlX75LB3MOuNk9parghy/view?usp=sharing
# https://drive.google.com/file/d/1-gSriHGICjnT7gETcfa7xtfwDqMep_mz/view?usp=sharing
# https://drive.google.com/file/d/1-BruVhjr5R6HG7hEt_gbEYmL5mCcUmB1/view?usp=sharing
# https://drive.google.com/file/d/1-CF5VzaIPBsdO-je9Vz6EJhV_EgXtdcm/view?usp=sharing