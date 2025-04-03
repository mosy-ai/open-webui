import pandas as pd
import json

def extract_images(mediafiles):
    if not mediafiles:
        return []
    media_dict = json.loads(mediafiles)
    return [img['mainImage'] for img in media_dict.get('images', [])]

def process_strapi_content(strapi_data):
    if not strapi_data:
        return {}
    
    data = json.loads(strapi_data)
    
    # Remove HTML tags from content
    content = data.get('content', '')
    if content:
        # Remove HTML tags but keep the text content
        import re
        content = re.sub(r'<[^>]+>', ' ', content)
        # Replace multiple spaces and newlines with single space
        content = re.sub(r'\s+', ' ', content)
        # Remove leading/trailing whitespace
        content = content.strip()
    else:
        content = None
    
    return {
        'content': content,
        'seo': data.get('seo', {})
    }

def create_metadata(row):
    return {
        'productName': row['productName'],
        'mainCategory': row['mainCategory'],
        'productCategory': json.loads(row['productCategory']),
        'price': row['price'],
        'designForm': row['designForm'],
        'images': extract_images(row['mediafiles'])
    }

def create_embedding_content(row, strapi_data):
    # Create a concise version for embedding
    metadata = create_metadata(row)
    seo = strapi_data.get('seo', {})
    
    if strapi_data.get('content'):
        embedding_content = f"{strapi_data.get('content')} {', '.join(metadata['productCategory'])}"
    else:
        embedding_content = f"""
        {metadata['productName']} {seo.get('title', '')} {seo.get('description', '')} {', '.join(metadata['productCategory'])} {metadata['designForm']}
        """
        
    return embedding_content.strip()

def create_content(row, strapi_data):
    metadata = create_metadata(row)
    seo = strapi_data.get('seo', {})
    
    if strapi_data.get('content'):
        content = f"""
        productName: {metadata['productName']}
        mainCategory: {metadata['mainCategory']}
        productCategory: {metadata['productCategory']}
        designForm: {metadata['designForm']}
        price: {metadata['price']} VND
        description: {strapi_data.get('content')}
        productImages: {metadata['images']}
        """
    else:
        content = f"""
        productName: {metadata['productName']}
        mainCategory: {metadata['mainCategory']}
        productCategory: {metadata['productCategory']}
        designForm: {metadata['designForm']}
        price: {metadata['price']} VND
        description: {seo.get('description', '')}
        productImages: {metadata['images']}
        """
        
    return content.strip()

# Read the CSV file
df = pd.read_csv("vcr_products.csv", sep=';')

# Create new dataframe with processed columns
processed_data = []

for index, row in df.iterrows():
    strapi_data = process_strapi_content(row['strapi'])
    
    processed_row = {
        'metadata': create_metadata(row),
        'embedding_content': create_embedding_content(row, strapi_data),
        'content': create_content(row, strapi_data)
    }
    for key, value in processed_row.items():
        print(f"{key}: {value}")
    print("-"*100)
    processed_data.append(processed_row)
    
print(len(processed_data))
    
# Create new dataframe and save to CSV
result_df = pd.DataFrame(processed_data)
result_df.to_csv('processed_products.csv', index=False)
print("Processing complete. Check processed_products.csv")