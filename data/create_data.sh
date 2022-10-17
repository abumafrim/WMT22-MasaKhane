repodir=../other-repos
datadir=../data

mkdir -p $repodir
cd $repodir

echo "Cloning supporting repositories: fairseq, lafand-mt and Web-Crawl-African"

repo=( "fairseq" "lafand-mt" "Web-Crawl-African" )
repourl=( "https://github.com/facebookresearch/fairseq.git" "https://github.com/masakhane-io/lafand-mt.git" "https://github.com/pavanpankaj/Web-Crawl-African.git" )

for i in "${!repo[@]}"; do
    if [ ! -d "${repo[i]}" ] ; then
        git clone ${repourl[i]} ${repo[i]}
    else
        cd "${repo[i]}"
        git pull ${repourl[i]}
        cd -
    fi
done

cd $datadir

echo "Preparing datasets"

pip install -r ../requirements.txt

#python scripts/mafand.py
python scripts/wmt22_african.py
#python scripts/opus_data.py
#python scripts/webcrawl_african.py
#python scripts/lava.py