# SYNTARA-PRO: Complete Examples Repository

## Examples Overview

This repository contains comprehensive examples for SYNTARA-PRO API usage:

### 📁 Examples Structure
```
examples/
├── basic/
│   ├── hello_world.py          # Simple text generation
│   ├── neural_processing.py     # Neural data processing
│   └── vision_analysis.py      # Image processing
├── advanced/
│   ├── streaming_client.py      # Real-time streaming
│   ├── batch_processing.py     # Batch requests
│   └── multimodal.py          # Multi-modal processing
├── bilingual/
│   ├── hindi_support.py        # Hindi text processing
│   ├── translation.py          # Translation between languages
│   └── code_switching.py      # Mixed language handling
├── production/
│   ├── error_handling.py       # Robust error handling
│   ├── rate_limiting.py       # Rate limiting implementation
│   └── monitoring.py          # Performance monitoring
└── integrations/
    ├── flask_app.py           # Flask integration
    ├── django_app.py          # Django integration
    └── react_client.py        # React frontend
```

---

## Basic Examples

### 1. Hello World (basic/hello_world.py)
```python
#!/usr/bin/env python3
"""
SYNTARA-PRO Basic Example: Hello World
Simple text generation example
"""

import requests
import json
import time

def main():
    # API endpoint
    url = "http://localhost:8000/process"
    
    # Request data
    payload = {
        "input_data": "Hello, SYNTARA-PRO! Tell me about your capabilities.",
        "task_type": "text_generation",
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    print("🚀 Sending request to SYNTARA-PRO...")
    print(f"📝 Input: {payload['input_data']}")
    
    try:
        # Send request
        start_time = time.time()
        response = requests.post(url, json=payload)
        processing_time = time.time() - start_time
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Success! Processing time: {processing_time:.2f}s")
            print(f"🤖 Response: {result['result']}")
            print(f"📊 Metadata: {result['metadata']}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")

if __name__ == "__main__":
    main()
```

### 2. Neural Processing (basic/neural_processing.py)
```python
#!/usr/bin/env python3
"""
SYNTARA-PRO Neural Processing Example
Process neural data through spiking networks
"""

import numpy as np
import requests
import matplotlib.pyplot as plt

def generate_neural_data():
    """Generate sample neural data."""
    # Simulate neural activity
    time_points = 1000
    channels = 64
    
    # Create realistic neural patterns
    data = np.random.randn(time_points, channels)
    
    # Add some structure (oscillations)
    for t in range(time_points):
        data[t] += 0.5 * np.sin(2 * np.pi * t / 100)
        data[t] += 0.3 * np.cos(2 * np.pi * t / 50)
    
    return data

def visualize_neural_data(data):
    """Visualize neural activity."""
    plt.figure(figsize=(12, 8))
    
    # Plot first few channels
    for i in range(min(8, data.shape[1])):
        plt.subplot(2, 4, i + 1)
        plt.plot(data[:, i], linewidth=0.5)
        plt.title(f"Channel {i+1}")
    plt.tight_layout()
    plt.savefig('neural_activity.png')
    plt.show()

def main():
    print("🧠 SYNTARA-PRO Neural Processing Example")
    
    # Generate neural data
    print("📊 Generating neural data...")
    neural_data = generate_neural_data()
    
    # Visualize
    print("📈 Visualizing neural activity...")
    visualize_neural_data(neural_data)
    
    # Process with SYNTARA-PRO
    print("🚀 Processing with SYNTARA-PRO...")
    
    url = "http://localhost:8000/process"
    payload = {
        "input_data": neural_data.tolist(),
        "task_type": "neural_processing"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Neural processing complete!")
            print(f"🧠 Modules used: {result['metadata']['modules_used']}")
            print(f"⏱️ Processing time: {result['metadata']['processing_time']:.3f}s")
            
            # Display results
            if 'spiking_stats' in result['result']:
                stats = result['result']['spiking_stats']
                print(f"🔥 Total spikes: {stats.get('total_spikes', 0)}")
                print(f"🧬 Neurons active: {stats.get('n_neurons', 0)}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

### 3. Vision Analysis (basic/vision_analysis.py)
```python
#!/usr/bin/env python3
"""
SYNTARA-PRO Vision Analysis Example
Process images with computer vision
"""

import requests
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

def create_test_image():
    """Create a test image for processing."""
    # Create a colorful test pattern
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Add geometric patterns
    for i in range(224):
        for j in range(224):
            # Create gradient
            img_array[i, j, 0] = int(255 * i / 224)  # Red gradient
            img_array[i, j, 1] = int(255 * j / 224)  # Green gradient
            img_array[i, j, 2] = int(255 * (i + j) / 448)  # Blue gradient
    
    # Add some shapes
    img_array[50:100, 50:100] = [255, 255, 255]  # White square
    img_array[150:200, 150:200] = [0, 0, 0]      # Black square
    
    return Image.fromarray(img_array)

def visualize_results(original_image, result):
    """Visualize vision processing results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Results visualization
    if result and 'success' in result and result['success']:
        # Create a simple visualization of features
        features = result.get('result', {}).get('features', [])
        if features:
            feature_array = np.array(features[:100]).reshape(10, 10)
            axes[1].imshow(feature_array, cmap='viridis')
            axes[1].set_title("Extracted Features")
        else:
            axes[1].text(0.5, 0.5, "No features extracted", 
                          ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title("Processing Result")
    else:
        axes[1].text(0.5, 0.5, "Processing Failed", 
                      ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("Error")
    
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('vision_analysis.png')
    plt.show()

def main():
    print("👁️ SYNTARA-PRO Vision Analysis Example")
    
    # Create test image
    print("🎨 Creating test image...")
    test_image = create_test_image()
    test_image.save('test_image.png')
    
    # Convert to bytes for API
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Process with SYNTARA-PRO
    print("🚀 Processing image with SYNTARA-PRO...")
    
    url = "http://localhost:8000/process"
    
    # Send as multipart form data
    files = {'image': ('test_image.jpg', img_bytes, 'image/jpeg')}
    data = {'task_type': 'vision'}
    
    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Vision processing complete!")
            print(f"👁️ Modules used: {result['metadata']['modules_used']}")
            print(f"⏱️ Processing time: {result['metadata']['processing_time']:.3f}s")
            
            # Display results
            vision_result = result.get('result', {})
            if 'num_patches' in vision_result:
                print(f"🔲 Image patches: {vision_result['num_patches']}")
            if 'logits' in vision_result:
                top_predictions = vision_result['logits'][:5]
                print(f"🏆 Top predictions: {top_predictions}")
            
            # Visualize
            visualize_results(test_image, result)
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

---

## Advanced Examples

### 4. Streaming Client (advanced/streaming_client.py)
```python
#!/usr/bin/env python3
"""
SYNTARA-PRO Streaming Example
Real-time streaming responses for long generations
"""

import requests
import json
import time
import threading
from queue import Queue

class StreamingClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.response_queue = Queue()
    
    def stream_process(self, input_data, max_tokens=500):
        """Process with streaming response."""
        url = f"{self.base_url}/process"
        
        payload = {
            "input_data": input_data,
            "task_type": "text_generation",
            "max_tokens": max_tokens,
            "stream": True
        }
        
        print(f"🚀 Starting streaming process...")
        print(f"📝 Input: {input_data[:50]}...")
        
        try:
            response = requests.post(url, json=payload, stream=True)
            
            if response.status_code == 200:
                full_response = ""
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            
                            if data_str == '[DONE]':
                                print("\n✅ Streaming complete!")
                                break
                            
                            try:
                                data = json.loads(data_str)
                                self.handle_stream_chunk(data)
                                
                                if data['type'] == 'complete':
                                    full_response = data['result']['result']
                                    
                            except json.JSONDecodeError:
                                continue
                
                return full_response
            else:
                print(f"❌ Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            return None
    
    def handle_stream_chunk(self, chunk):
        """Handle individual streaming chunk."""
        chunk_type = chunk.get('type', 'unknown')
        
        if chunk_type == 'start':
            print("🎬 Processing started...")
            
        elif chunk_type == 'progress':
            step = chunk.get('step', 0)
            total = chunk.get('total', 1)
            message = chunk.get('message', '')
            print(f"⏳ Progress: {step}/{total} - {message}")
            
        elif chunk_type == 'complete':
            result = chunk.get('result', {})
            print(f"✅ Final result: {result.get('result', '')[:100]}...")
            
        elif chunk_type == 'error':
            error = chunk.get('error', 'Unknown error')
            print(f"❌ Error: {error}")

def main():
    print("🌊 SYNTARA-PRO Streaming Example")
    
    client = StreamingClient()
    
    # Example 1: Long text generation
    print("\n" + "="*50)
    print("Example 1: Long Text Generation")
    print("="*50)
    
    long_prompt = """
    Write a detailed explanation of artificial intelligence, covering:
    1. History and evolution
    2. Current technologies and approaches
    3. Applications in various fields
    4. Future prospects and challenges
    5. Ethical considerations
    
    Make it comprehensive and educational.
    """
    
    result = client.stream_process(long_prompt, max_tokens=800)
    
    if result:
        print(f"\n📄 Full response length: {len(result)} characters")
        print(f"📝 First 200 chars: {result[:200]}...")
    
    # Example 2: Creative writing
    print("\n" + "="*50)
    print("Example 2: Creative Writing")
    print("="*50)
    
    creative_prompt = """
    Write a short science fiction story about:
    - An AI that becomes self-aware
    - Set in the year 2045
    - Include themes of consciousness and identity
    - Make it thought-provoking
    - Around 500 words
    """
    
    result = client.stream_process(creative_prompt, max_tokens=600)
    
    if result:
        print(f"\n📖 Story length: {len(result)} words")
        print(f"✨ First 200 chars: {result[:200]}...")

if __name__ == "__main__":
    main()
```

### 5. Batch Processing (advanced/batch_processing.py)
```python
#!/usr/bin/env python3
"""
SYNTARA-PRO Batch Processing Example
Process multiple requests efficiently
"""

import requests
import time
import concurrent.futures
from typing import List, Dict

class BatchProcessor:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def process_batch(self, requests: List[Dict]) -> List[Dict]:
        """Process multiple requests in batch."""
        url = f"{self.base_url}/batch"
        
        print(f"🚀 Processing batch of {len(requests)} requests...")
        
        try:
            response = requests.post(url, json={"requests": requests})
            
            if response.status_code == 200:
                result = response.json()
                return result.get('results', [])
            else:
                print(f"❌ Batch error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            return []
    
    def process_parallel(self, requests: List[Dict], max_workers: int = 5) -> List[Dict]:
        """Process requests in parallel."""
        url = f"{self.base_url}/process"
        
        def single_request(request_data):
            try:
                response = requests.post(url, json=request_data, timeout=30)
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"success": False, "error": f"HTTP {response.status_code}"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        print(f"🔄 Processing {len(requests)} requests with {max_workers} workers...")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_request, req) for req in requests]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        processing_time = time.time() - start_time
        
        print(f"✅ Parallel processing complete in {processing_time:.2f}s")
        
        return results

def create_test_requests() -> List[Dict]:
    """Create diverse test requests."""
    requests = [
        # Text generation
        {
            "input_data": "What is machine learning?",
            "task_type": "text_generation",
            "max_tokens": 100
        },
        {
            "input_data": "Explain quantum computing",
            "task_type": "text_generation",
            "max_tokens": 150
        },
        
        # Neural processing
        {
            "input_data": [1.0, 2.0, 3.0, 4.0, 5.0] * 200,
            "task_type": "neural_processing"
        },
        {
            "input_data": [0.5, -0.3, 0.8, -0.2] * 250,
            "task_type": "neural_processing"
        },
        
        # Creative tasks
        {
            "input_data": "Write a haiku about technology",
            "task_type": "text_generation",
            "max_tokens": 50,
            "temperature": 0.9
        },
        {
            "input_data": "Create a slogan for an AI company",
            "task_type": "text_generation",
            "max_tokens": 30,
            "temperature": 0.8
        },
        
        # Analysis tasks
        {
            "input_data": "Analyze the sentiment: 'I love this product!'",
            "task_type": "text_analysis"
        },
        {
            "input_data": "Summarize: AI is transforming healthcare",
            "task_type": "text_analysis"
        }
    ]
    
    return requests

def analyze_results(results: List[Dict]):
    """Analyze processing results."""
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r.get('success', False))
    failed_requests = total_requests - successful_requests
    
    print(f"\n📊 Batch Analysis:")
    print(f"   Total requests: {total_requests}")
    print(f"   ✅ Successful: {successful_requests}")
    print(f"   ❌ Failed: {failed_requests}")
    print(f"   📈 Success rate: {successful_requests/total_requests*100:.1f}%")
    
    # Analyze processing times
    processing_times = [
        r.get('metadata', {}).get('processing_time', 0) 
        for r in results if r.get('success', False)
    ]
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        
        print(f"\n⏱️ Processing Times:")
        print(f"   Average: {avg_time:.3f}s")
        print(f"   Minimum: {min_time:.3f}s")
        print(f"   Maximum: {max_time:.3f}s")
    
    # Analyze modules used
    all_modules = []
    for r in results:
        modules = r.get('metadata', {}).get('modules_used', [])
        all_modules.extend(modules)
    
    if all_modules:
        from collections import Counter
        module_counts = Counter(all_modules)
        
        print(f"\n🔧 Modules Used:")
        for module, count in module_counts.most_common():
            print(f"   {module}: {count} times")

def main():
    print("📦 SYNTARA-PRO Batch Processing Example")
    
    # Create test requests
    test_requests = create_test_requests()
    
    processor = BatchProcessor()
    
    # Method 1: Batch endpoint
    print("\n" + "="*50)
    print("Method 1: Batch Endpoint")
    print("="*50)
    
    batch_results = processor.process_batch(test_requests)
    analyze_results(batch_results)
    
    # Method 2: Parallel processing
    print("\n" + "="*50)
    print("Method 2: Parallel Processing")
    print("="*50)
    
    parallel_results = processor.process_parallel(test_requests, max_workers=3)
    analyze_results(parallel_results)
    
    # Comparison
    print("\n" + "="*50)
    print("Comparison")
    print("="*50)
    
    batch_success = sum(1 for r in batch_results if r.get('success', False))
    parallel_success = sum(1 for r in parallel_results if r.get('success', False))
    
    print(f"Batch endpoint success rate: {batch_success/len(test_requests)*100:.1f}%")
    print(f"Parallel processing success rate: {parallel_success/len(test_requests)*100:.1f}%")

if __name__ == "__main__":
    main()
```

---

## Bilingual Examples

### 6. Hindi Support (bilingual/hindi_support.py)
```python
#!/usr/bin/env python3
"""
SYNTARA-PRO Hindi Language Support Example
Demonstrate Hindi text processing capabilities
"""

import requests
import json

def hindi_text_examples():
    """Examples of Hindi text processing."""
    return [
        {
            "name": "Basic Greeting",
            "text": "नमस्ते दुनिया! मैं SYNTARA-PRO हूँ।",
            "expected_type": "text_generation"
        },
        {
            "name": "Question",
            "text": "कृपष्णा बताइए कि आर्टिफिशियल इंटेलिजेंस क्या है?",
            "expected_type": "text_generation"
        },
        {
            "name": "Creative Writing",
            "text": "एक कहानी लिखें जो भविष्य के भविष्य में समाप्त हो",
            "expected_type": "text_generation",
            "temperature": 0.8,
            "max_tokens": 200
        },
        {
            "name": "Technical Explanation",
            "text": "मशीन लर्निंग की बुनियादी समझाइए",
            "expected_type": "text_generation",
            "max_tokens": 300
        },
        {
            "name": "Poetry",
            "text": "प्रेम और तकनीक पर एक कविता लिखें",
            "expected_type": "text_generation",
            "temperature": 0.9,
            "max_tokens": 150
        }
    ]

def process_hindi_text(text, task_type="text_generation", **kwargs):
    """Process Hindi text with SYNTARA-PRO."""
    url = "http://localhost:8000/process"
    
    payload = {
        "input_data": text,
        "task_type": task_type,
        **kwargs
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    print("🇮🇳 SYNTARA-PRO Hindi Language Support Example")
    print("="*60)
    
    examples = hindi_text_examples()
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print("-" * 40)
        print(f"📝 Input: {example['text']}")
        
        # Process the text
        result = process_hindi_text(
            example['text'],
            example['expected_type'],
            **{k: v for k, v in example.items() 
               if k not in ['name', 'text', 'expected_type']}
        )
        
        if result.get('success', False):
            print(f"✅ Success!")
            print(f"🤖 Response: {result.get('result', '')[:200]}...")
            print(f"🔧 Modules: {result.get('metadata', {}).get('modules_used', [])}")
            print(f"⏱️ Time: {result.get('metadata', {}).get('processing_time', 0):.3f}s")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        print()

def mixed_language_example():
    """Example of mixed language processing."""
    print("\n" + "="*60)
    print("Mixed Language Example (Hinglish)")
    print("="*60)
    
    mixed_texts = [
        "Hello दोस्तों, how are you aaj kal?",
        "Mujhe AI ke baare mein jaanna hai",
        "Technology बहुत ज़रूरी badh rahi hai",
        "Let's discuss about भविष्य का future"
    ]
    
    for text in mixed_texts:
        print(f"\n📝 Mixed: {text}")
        result = process_hindi_text(text, "text_generation")
        
        if result.get('success', False):
            print(f"🤖 Response: {result.get('result', '')[:100]}...")

if __name__ == "__main__":
    main()
    mixed_language_example()
```

---

## Production Examples

### 7. Error Handling (production/error_handling.py)
```python
#!/usr/bin/env python3
"""
SYNTARA-PRO Production Error Handling Example
Robust error handling and retry logic
"""

import requests
import time
import json
import logging
from typing import Optional, Dict, Any
from enum import Enum

class ErrorType(Enum):
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"

class SyntaraProClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def make_request(self, endpoint: str, payload: Dict, 
                   retries: int = None) -> Optional[Dict]:
        """Make robust API request with retry logic."""
        if retries is None:
            retries = self.max_retries
        
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        for attempt in range(retries + 1):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{retries + 1}: {endpoint}")
                
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                # Handle different HTTP status codes
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    self.logger.warning("Rate limit hit, implementing backoff")
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                elif response.status_code >= 500:
                    self.logger.warning(f"Server error: {response.status_code}")
                    if attempt < retries:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        return self.create_error_response(
                            ErrorType.SERVER_ERROR,
                            f"Server error after {retries} retries"
                        )
                else:
                    return self.create_error_response(
                        ErrorType.API_ERROR,
                        f"HTTP {response.status_code}: {response.text}"
                    )
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    return self.create_error_response(
                        ErrorType.TIMEOUT,
                        f"Timeout after {retries} retries"
                    )
                    
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"Connection error on attempt {attempt + 1}")
                if attempt < retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    return self.create_error_response(
                        ErrorType.NETWORK_ERROR,
                        f"Connection failed after {retries} retries"
                    )
                    
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                return self.create_error_response(
                    ErrorType.API_ERROR,
                    f"Unexpected error: {str(e)}"
                )
        
        return None
    
    def create_error_response(self, error_type: ErrorType, message: str) -> Dict:
        """Create standardized error response."""
        return {
            "success": False,
            "error": message,
            "error_type": error_type.value,
            "timestamp": time.time()
        }
    
    def process_with_fallback(self, input_data: str, 
                           primary_task: str = "text_generation",
                           fallback_task: str = "text_analysis") -> Dict:
        """Process with fallback task type."""
        
        # Try primary task
        result = self.make_request("/process", {
            "input_data": input_data,
            "task_type": primary_task
        })
        
        if result and result.get('success', False):
            return result
        
        # Try fallback task
        self.logger.info(f"Primary task failed, trying fallback: {fallback_task}")
        fallback_result = self.make_request("/process", {
            "input_data": input_data,
            "task_type": fallback_task
        })
        
        if fallback_result and fallback_result.get('success', False):
            fallback_result['fallback_used'] = True
            fallback_result['original_task'] = primary_task
            return fallback_result
        
        # Both failed
        return self.create_error_response(
            ErrorType.API_ERROR,
            f"Both primary and fallback tasks failed"
        )

def demonstrate_error_handling():
    """Demonstrate various error scenarios."""
    client = SyntaraProClient(api_key="test-key")
    
    print("🛡️ SYNTARA-PRO Error Handling Examples")
    print("="*50)
    
    # Example 1: Normal request
    print("\n1. Normal Request")
    result = client.make_request("/process", {
        "input_data": "Hello world",
        "task_type": "text_generation"
    })
    
    if result:
        print(f"✅ Success: {result.get('success', False)}")
        if not result.get('success', False):
            print(f"❌ Error: {result.get('error', 'Unknown')}")
    
    # Example 2: Invalid endpoint
    print("\n2. Invalid Endpoint")
    result = client.make_request("/invalid", {
        "input_data": "test"
    })
    
    if result:
        print(f"✅ Success: {result.get('success', False)}")
        if not result.get('success', False):
            print(f"❌ Error: {result.get('error', 'Unknown')}")
    
    # Example 3: Fallback mechanism
    print("\n3. Fallback Mechanism")
    result = client.process_with_fallback(
        "Complex input that might fail",
        primary_task="invalid_task",
        fallback_task="text_analysis"
    )
    
    if result:
        print(f"✅ Success: {result.get('success', False)}")
        print(f"🔄 Fallback used: {result.get('fallback_used', False)}")
        if result.get('success', False):
            print(f"❌ Error: {result.get('error', 'Unknown')}")

def main():
    demonstrate_error_handling()

if __name__ == "__main__":
    main()
```

---

## Running Examples

### Prerequisites
```bash
# Install required packages
pip install requests numpy matplotlib pillow

# Start SYNTARA-PRO server
python syntara_pro_server.py
```

### Run Examples
```bash
# Basic examples
python basic/hello_world.py
python basic/neural_processing.py
python basic/vision_analysis.py

# Advanced examples
python advanced/streaming_client.py
python advanced/batch_processing.py

# Bilingual examples
python bilingual/hindi_support.py

# Production examples
python production/error_handling.py
```

---

## Expected Output

Each example will demonstrate different aspects of SYNTARA-PRO:

1. **Hello World**: Basic text generation
2. **Neural Processing**: Spiking network analysis
3. **Vision Analysis**: Image feature extraction
4. **Streaming**: Real-time response streaming
5. **Batch Processing**: Efficient bulk processing
6. **Hindi Support**: Multilingual capabilities
7. **Error Handling**: Robust production code

---

*For more examples, check the examples directory!*
