from DeepImageSearch import Load_Data,Search_Setup
import Part3

# Load images from a folder
image_list = Load_Data().from_folder(['C:/Users/njain/Downloads/images'])
image_to_verify = 'C:/Users/njain/Downloads/checkA.jpg'
print("Total Image Count:",len(image_list))
print("Samples:")
print(image_list[:10])

# Set up the search engine
st = Search_Setup(image_list=image_list,model_name='vgg19',pretrained=True,image_count=100)
# Index the images
st.run_index()

# Get metadata
metadata = st.get_image_metadata_file()
print(metadata)

# Add new images to the index
st.add_images_to_index(image_list[1001:1010])

# Get similar images
img1=st.get_similar_images(image_to_verify,number_of_images=5)
values= list(img1.values())


# Test the function
percentage=Part3.calculateSimilarity(values[0],image_to_verify)
finalimage=values[0]
for str in values:
    inter_val=Part3.calculateSimilarity(str, image_to_verify)
    if percentage<inter_val :
        percentage=inter_val
        finalimage=str


if percentage>60:
    print("Matching image found is ", finalimage)
    print("With Matching Percentage ",percentage)
else:
    print("Sorry No Match Found")
