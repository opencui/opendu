tag=build-image-$(date +'%Y%m%d%H%M%S')
git tag $tag
git push origin $tag
