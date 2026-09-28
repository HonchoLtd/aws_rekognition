# Prerequisite:

```
go get github.com/aws/aws-sdk-go-v2/aws
go get github.com/aws/aws-sdk-go-v2/service/rekognition
go get github.com/aws/aws-sdk-go-v2/config
go get github.com/samber/lo
```

Call This in converter to index face from image upload, imageID that will be return in `SearchFace` So make sure put correct id so you can easy find in gallery later on
```
IndexFace(ctx context.Context, image []byte, imageID string, eventID string) error
```

Call this in selfie to get face from selfie and search image in from eventID
```
SearchFace(ctx context.Context, imageSelfie []byte, eventID string) ([]string, error)
```

Make sure the eventID is same, or we can't make correct collections.


<h3>facev2 package</h3>

```go
type Face interface {
    IndexFace(ctx context.Context, image []byte, externalImageId string, collectionId string) ([]IndexedFace, error)
    IndexFaceWithBucket(ctx context.Context, s3Bucket string, s3Key string, externalImageId string, collectionId string) ([]IndexedFace, error)
    SearchFacesByFaceId(ctx context.Context, collectionId string, faceId string, opts ...SearchFacesOption) ([]FaceMatch, error)
}
```
Look inside the [facev2 documentation](facev2/README.md)