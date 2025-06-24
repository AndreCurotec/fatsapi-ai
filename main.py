from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import openai
import os
from dotenv import load_dotenv
import json
import math
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(title="Chatbot Demo API")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # For local development
        "https://*.netlify.app",  # For Netlify deployment
        "https://shimmering-crostata-d31ea8.netlify.app"  # Replace with your actual Netlify URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Request/Response models
class ChatMessage(BaseModel):
    message: str
    category: str
    conversation_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str
    is_complete: bool = False
    user_profile: Optional[Dict[str, Any]] = None
    provider_types: Optional[List[Dict[str, Any]]] = None

class CategorySelection(BaseModel):
    category: str

# Mock Provider Database
PROVIDERS_DB = {
    "plan_protect": [
        # Elder Law Attorneys
        {
            "id": 1,
            "name": "Wilson Elder Law Firm",
            "type": "Elder Law Attorney",
            "specialties": ["estate planning", "wills", "trusts", "asset protection"],
            "rating": 4.8,
            "location": "Downtown",
            "description": "Specializing in comprehensive estate planning and asset protection for seniors.",
            "expertise_vector": [0.9, 0.1, 0.8, 0.3, 0.7]
        },
        {
            "id": 2,
            "name": "Heritage Legal Partners",
            "type": "Elder Law Attorney",
            "specialties": ["estate planning", "probate", "guardianship", "medicaid planning"],
            "rating": 4.9,
            "location": "Westside",
            "description": "Premier elder law practice focused on protecting families through life transitions.",
            "expertise_vector": [0.8, 0.2, 0.9, 0.4, 0.6]
        },
        {
            "id": 3,
            "name": "Goldstein Estate Law",
            "type": "Elder Law Attorney",
            "specialties": ["wills", "trusts", "tax planning", "business succession"],
            "rating": 4.7,
            "location": "Financial District",
            "description": "Experienced attorneys specializing in sophisticated estate and tax planning.",
            "expertise_vector": [0.9, 0.1, 0.7, 0.5, 0.8]
        },
        
        # Financial Planners
        {
            "id": 4,
            "name": "SecureFuture Financial",
            "type": "Financial Planner",
            "specialties": ["retirement planning", "insurance", "long-term care", "investments"],
            "rating": 4.6,
            "location": "Midtown",
            "description": "Expert financial planning for retirement and long-term care needs.",
            "expertise_vector": [0.8, 0.2, 0.6, 0.9, 0.5]
        },
        {
            "id": 5,
            "name": "Meridian Wealth Advisors",
            "type": "Financial Planner",
            "specialties": ["retirement planning", "investment management", "tax strategies", "estate coordination"],
            "rating": 4.8,
            "location": "Uptown",
            "description": "Comprehensive wealth management focusing on retirement security and legacy planning.",
            "expertise_vector": [0.7, 0.3, 0.8, 0.8, 0.6]
        },
        {
            "id": 6,
            "name": "Pinnacle Planning Group",
            "type": "Financial Planner",
            "specialties": ["financial planning", "insurance analysis", "retirement income", "healthcare costs"],
            "rating": 4.5,
            "location": "Southside",
            "description": "Holistic financial planning with emphasis on healthcare and long-term care preparation.",
            "expertise_vector": [0.6, 0.4, 0.7, 0.9, 0.7]
        },
        
        # Insurance Specialists
        {
            "id": 7,
            "name": "Guardian Insurance Services",
            "type": "Insurance Specialist",
            "specialties": ["life insurance", "disability insurance", "long-term care insurance"],
            "rating": 4.7,
            "location": "Downtown",
            "description": "Comprehensive insurance solutions for life and health protection.",
            "expertise_vector": [0.6, 0.1, 0.4, 0.8, 0.9]
        },
        {
            "id": 8,
            "name": "ProtectLife Insurance Brokers",
            "type": "Insurance Specialist",
            "specialties": ["long-term care insurance", "life insurance", "annuities", "medicare supplements"],
            "rating": 4.6,
            "location": "Eastside",
            "description": "Independent insurance brokers specializing in senior protection products.",
            "expertise_vector": [0.5, 0.2, 0.5, 0.9, 0.8]
        },
        {
            "id": 9,
            "name": "Secure Horizons Insurance",
            "type": "Insurance Specialist",
            "specialties": ["disability insurance", "life insurance", "critical illness", "income protection"],
            "rating": 4.4,
            "location": "Northside",
            "description": "Specialized in income and health protection insurance for working professionals.",
            "expertise_vector": [0.7, 0.1, 0.3, 0.7, 0.9]
        }
    ],
    "care_aging": [
        # Geriatric Care Managers
        {
            "id": 10,
            "name": "CompassCare Management",
            "type": "Geriatric Care Manager",
            "specialties": ["care coordination", "health advocacy", "family support", "crisis management"],
            "rating": 4.9,
            "location": "Central",
            "description": "Professional care coordination and advocacy for aging adults and families.",
            "expertise_vector": [0.2, 0.9, 0.8, 0.4, 0.6]
        },
        {
            "id": 11,
            "name": "ElderCare Navigators",
            "type": "Geriatric Care Manager",
            "specialties": ["care planning", "resource coordination", "medical advocacy", "family consultation"],
            "rating": 4.8,
            "location": "Downtown",
            "description": "Expert navigation through the complex world of aging services and healthcare.",
            "expertise_vector": [0.3, 0.8, 0.9, 0.5, 0.5]
        },
        {
            "id": 12,
            "name": "Caring Connections",
            "type": "Geriatric Care Manager",
            "specialties": ["geriatric assessment", "care coordination", "discharge planning", "long-term care"],
            "rating": 4.7,
            "location": "Westside",
            "description": "Comprehensive geriatric care management focusing on seamless care transitions.",
            "expertise_vector": [0.4, 0.9, 0.7, 0.6, 0.4]
        },
        
        # Home Care Agencies
        {
            "id": 13,
            "name": "HomeHeart Care Services",
            "type": "Home Care Agency",
            "specialties": ["personal care", "medication management", "companionship", "meal preparation"],
            "rating": 4.7,
            "location": "Westside",
            "description": "Compassionate in-home care services for daily living assistance.",
            "expertise_vector": [0.1, 0.8, 0.9, 0.3, 0.4]
        },
        {
            "id": 14,
            "name": "Golden Years Home Care",
            "type": "Home Care Agency",
            "specialties": ["personal care", "housekeeping", "transportation", "respite care"],
            "rating": 4.6,
            "location": "Eastside",
            "description": "Reliable home care services helping seniors maintain independence and dignity.",
            "expertise_vector": [0.2, 0.7, 0.8, 0.4, 0.5]
        },
        {
            "id": 15,
            "name": "Comfort Keepers",
            "type": "Home Care Agency",
            "specialties": ["companionship", "personal care", "medication reminders", "safety monitoring"],
            "rating": 4.5,
            "location": "Southside",
            "description": "Uplifting in-home care that helps seniors live safely and independently.",
            "expertise_vector": [0.3, 0.6, 0.9, 0.2, 0.6]
        },
        
        # Senior Communities
        {
            "id": 16,
            "name": "Sunrise Senior Living",
            "type": "Senior Community",
            "specialties": ["assisted living", "memory care", "independent living", "rehabilitation"],
            "rating": 4.5,
            "location": "Northside",
            "description": "Full-service senior living community with multiple care levels.",
            "expertise_vector": [0.3, 0.7, 0.8, 0.2, 0.7]
        },
        {
            "id": 17,
            "name": "Brookdale Senior Living",
            "type": "Senior Community",
            "specialties": ["independent living", "assisted living", "memory care", "skilled nursing"],
            "rating": 4.4,
            "location": "Midtown",
            "description": "Comprehensive senior living with personalized care and vibrant community life.",
            "expertise_vector": [0.4, 0.6, 0.7, 0.3, 0.8]
        },
        {
            "id": 18,
            "name": "Atria Senior Living",
            "type": "Senior Community",
            "specialties": ["assisted living", "memory care", "respite care", "life enrichment"],
            "rating": 4.3,
            "location": "Uptown",
            "description": "Engaging senior community focused on wellness, purpose, and connection.",
            "expertise_vector": [0.5, 0.5, 0.6, 0.4, 0.9]
        }
    ],
    "support_death": [
        # Hospice Care
        {
            "id": 19,
            "name": "Peaceful Transitions Hospice",
            "type": "Hospice Care",
            "specialties": ["end-of-life care", "pain management", "family support", "spiritual care"],
            "rating": 4.9,
            "location": "Downtown",
            "description": "Compassionate hospice care focused on comfort and dignity.",
            "expertise_vector": [0.1, 0.3, 0.2, 0.9, 0.8]
        },
        {
            "id": 20,
            "name": "Comfort Care Hospice",
            "type": "Hospice Care",
            "specialties": ["palliative care", "pain management", "bereavement support", "respite care"],
            "rating": 4.8,
            "location": "Westside",
            "description": "Comprehensive hospice and palliative care services with 24/7 support.",
            "expertise_vector": [0.2, 0.4, 0.3, 0.8, 0.7]
        },
        {
            "id": 21,
            "name": "Serenity Hospice Services",
            "type": "Hospice Care",
            "specialties": ["home hospice", "inpatient care", "grief counseling", "memorial services"],
            "rating": 4.7,
            "location": "Northside",
            "description": "Holistic hospice care emphasizing dignity, comfort, and family involvement.",
            "expertise_vector": [0.3, 0.2, 0.4, 0.9, 0.6]
        },
        
        # Estate Settlement
        {
            "id": 22,
            "name": "Legacy Estate Services",
            "type": "Estate Settlement",
            "specialties": ["probate", "estate administration", "asset distribution", "tax planning"],
            "rating": 4.6,
            "location": "Financial District",
            "description": "Expert guidance through estate settlement and probate processes.",
            "expertise_vector": [0.8, 0.2, 0.3, 0.7, 0.6]
        },
        {
            "id": 23,
            "name": "Heritage Estate Solutions",
            "type": "Estate Settlement",
            "specialties": ["estate administration", "probate court", "trust administration", "asset valuation"],
            "rating": 4.5,
            "location": "Midtown",
            "description": "Professional estate settlement services with focus on efficient administration.",
            "expertise_vector": [0.7, 0.3, 0.4, 0.6, 0.5]
        },
        {
            "id": 24,
            "name": "Meridian Probate Services",
            "type": "Estate Settlement",
            "specialties": ["probate", "estate planning", "tax preparation", "asset liquidation"],
            "rating": 4.4,
            "location": "Eastside",
            "description": "Streamlined probate and estate administration with personalized attention.",
            "expertise_vector": [0.9, 0.1, 0.2, 0.8, 0.4]
        },
        
        # Funeral Services
        {
            "id": 25,
            "name": "Grace Funeral Home",
            "type": "Funeral Services",
            "specialties": ["funeral planning", "cremation", "burial services", "grief counseling"],
            "rating": 4.8,
            "location": "Memorial District",
            "description": "Dignified funeral services with personalized care and support.",
            "expertise_vector": [0.2, 0.4, 0.1, 0.8, 0.9]
        },
        {
            "id": 26,
            "name": "Eternal Rest Funeral Chapel",
            "type": "Funeral Services",
            "specialties": ["traditional services", "cremation", "memorial planning", "pre-planning"],
            "rating": 4.7,
            "location": "Uptown",
            "description": "Family-owned funeral home providing compassionate service for over 50 years.",
            "expertise_vector": [0.3, 0.3, 0.2, 0.7, 0.8]
        },
        {
            "id": 27,
            "name": "Celebration of Life Services",
            "type": "Funeral Services",
            "specialties": ["celebration services", "cremation", "green burial", "life tributes"],
            "rating": 4.6,
            "location": "Southside",
            "description": "Modern funeral services focused on celebrating life and healing families.",
            "expertise_vector": [0.1, 0.5, 0.3, 0.6, 0.7]
        }
    ]
}

# System prompts for different categories
SYSTEM_PROMPTS = {
    "plan_protect": """You are a helpful assistant for Chatbot Demo, specializing in "Plan & Protect" services. 
    Your job is to understand the user's needs for estate planning, financial planning, insurance, or asset protection.
    Ask targeted questions to understand their situation, age, family status, assets, and concerns.
    Keep responses conversational and empathetic. After 3-4 exchanges, summarize their needs.""",
    
    "care_aging": """You are a helpful assistant for Chatbot Demo, specializing in "Care throughout Aging" services.
    Your job is to understand the user's needs for aging care, whether for themselves or a loved one.
    Ask about the person's age, current health, living situation, family support, and specific care needs.
    Keep responses warm and supportive. After 3-4 exchanges, summarize their care requirements.""",
    
    "support_death": """You are a helpful assistant for Chatbot Demo, specializing in "Support upon Death" services.
    Your job is to sensitively understand the user's needs related to end-of-life planning or bereavement support.
    Ask gentle questions about their situation, timeline, family needs, and what type of support they're seeking.
    Be especially compassionate and respectful. After 3-4 exchanges, summarize their support needs."""
}

async def get_openai_response(messages: List[Dict[str, str]], category: str) -> str:
    """Get response from OpenAI with category-specific system prompt"""
    try:
        system_message = {"role": "system", "content": SYSTEM_PROMPTS[category]}
        full_messages = [system_message] + messages
        
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=full_messages,
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector for text using OpenAI"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

def extract_user_needs(conversation_history: List[Dict[str, str]]) -> str:
    """Extract user needs from conversation history"""
    user_messages = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
    return " ".join(user_messages)

def get_provider_types(user_needs: str, category: str) -> List[Dict[str, Any]]:
    """Get recommended provider types based on user needs using sophisticated vector matching"""
    try:
        # Get category mapping
        category_map = {
            "plan_protect": "plan_protect",
            "care_aging": "care_aging", 
            "support_death": "support_death"
        }
        
        db_category = category_map.get(category, category)
        providers = PROVIDERS_DB.get(db_category, [])
        
        if not providers:
            return []
        
        # Create user needs vector
        user_vector = create_user_needs_vector(user_needs, category)
        
        # Group providers by type and calculate relevance using enhanced algorithm
        provider_types = {}
        user_words = set(user_needs.lower().split())
        
        for provider in providers:
            provider_type = provider["type"]
            
            if provider_type not in provider_types:
                # 1. Keyword similarity (Jaccard)
                specialty_words = set(" ".join(provider["specialties"]).lower().split())
                desc_words = set(provider["description"].lower().split())
                provider_words = specialty_words.union(desc_words)
                
                intersection = len(user_words.intersection(provider_words))
                union = len(user_words.union(provider_words))
                keyword_similarity = intersection / union if union > 0 else 0
                
                # 2. Vector similarity (Cosine) - average for all providers of this type
                provider_vector = provider["expertise_vector"]
                vector_similarity = cosine_similarity(user_vector, provider_vector)
                
                # 3. Specialty relevance boost
                specialty_boost = 0
                for word in user_words:
                    for specialty in provider["specialties"]:
                        if word in specialty.lower():
                            specialty_boost += 0.1
                specialty_boost = min(0.3, specialty_boost)  # Cap at 0.3
                
                # Combined relevance score: 40% keyword + 50% vector + 10% specialty boost
                relevance_score = (keyword_similarity * 0.4) + (vector_similarity * 0.5) + specialty_boost
                
                provider_types[provider_type] = {
                    "type": provider_type,
                    "description": provider["description"],
                    "relevance_score": relevance_score,
                    "provider_count": 1,
                    "specialties": provider["specialties"][:3],  # Show top 3 specialties
                    "vector_similarity": vector_similarity,
                    "keyword_similarity": keyword_similarity,
                    "vector_scores": [vector_similarity]  # Track individual scores for averaging
                }
            else:
                # Average the vector similarities for multiple providers of same type
                provider_vector = provider["expertise_vector"]
                vector_similarity = cosine_similarity(user_vector, provider_vector)
                provider_types[provider_type]["vector_scores"].append(vector_similarity)
                
                # Recalculate average vector similarity
                avg_vector_sim = sum(provider_types[provider_type]["vector_scores"]) / len(provider_types[provider_type]["vector_scores"])
                
                # Update relevance score with averaged vector similarity
                keyword_sim = provider_types[provider_type]["keyword_similarity"]
                specialty_boost = 0  # Already calculated for the type
                for word in user_words:
                    for specialty in provider["specialties"]:
                        if word in specialty.lower():
                            specialty_boost += 0.1
                specialty_boost = min(0.3, specialty_boost)
                
                provider_types[provider_type]["relevance_score"] = (keyword_sim * 0.4) + (avg_vector_sim * 0.5) + specialty_boost
                provider_types[provider_type]["vector_similarity"] = avg_vector_sim
                provider_types[provider_type]["provider_count"] += 1
        
        # Clean up temporary vector_scores field and convert to list
        result = []
        for provider_type in provider_types.values():
            del provider_type["vector_scores"]  # Remove temporary field
            result.append(provider_type)
        
        # Sort by relevance score
        result.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return result
        
    except Exception as e:
        print(f"Provider types error: {e}")
        return []

def create_user_needs_vector(user_needs: str, category: str) -> List[float]:
    """Create a user needs vector based on category and conversation content"""
    # Define semantic dimensions for matching
    # [planning_focus, care_intensity, urgency, emotional_support, technical_complexity]
    
    user_needs_lower = user_needs.lower()
    
    # Base category vectors
    category_vectors = {
        "plan_protect": [0.8, 0.2, 0.4, 0.3, 0.7],      # High planning, low care, medium urgency
        "care_aging": [0.3, 0.9, 0.6, 0.7, 0.4],        # Low planning, high care, high emotional
        "support_death": [0.4, 0.7, 0.8, 0.9, 0.5]      # Medium planning, high emotional, urgent
    }
    
    base_vector = category_vectors.get(category, [0.5, 0.5, 0.5, 0.5, 0.5])
    
    # Adjust based on user needs keywords
    keywords_map = {
        # Planning focus indicators
        "plan": (0, 0.3), "future": (0, 0.2), "estate": (0, 0.4), "will": (0, 0.3),
        "financial": (0, 0.3), "investment": (0, 0.2), "insurance": (0, 0.3),
        
        # Care intensity indicators  
        "care": (1, 0.3), "help": (1, 0.2), "assistance": (1, 0.3), "support": (1, 0.2),
        "daily": (1, 0.3), "living": (1, 0.2), "independent": (1, -0.2), "home": (1, 0.3),
        
        # Urgency indicators
        "urgent": (2, 0.4), "immediate": (2, 0.4), "soon": (2, 0.3), "now": (2, 0.3),
        "emergency": (2, 0.5), "quickly": (2, 0.3), "asap": (2, 0.4),
        
        # Emotional support indicators
        "difficult": (3, 0.3), "worried": (3, 0.3), "scared": (3, 0.3), "confused": (3, 0.2),
        "overwhelmed": (3, 0.4), "family": (3, 0.2), "emotional": (3, 0.4), "grief": (3, 0.5),
        
        # Technical complexity indicators
        "complex": (4, 0.3), "legal": (4, 0.4), "tax": (4, 0.4), "medical": (4, 0.3),
        "specialist": (4, 0.3), "expert": (4, 0.2), "professional": (4, 0.2)
    }
    
    # Apply keyword adjustments
    adjusted_vector = base_vector.copy()
    for keyword, (dimension, weight) in keywords_map.items():
        if keyword in user_needs_lower:
            adjusted_vector[dimension] = min(1.0, max(0.0, adjusted_vector[dimension] + weight))
    
    return adjusted_vector

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)

def match_providers(user_needs: str, category: str, provider_type: str = None) -> List[Dict[str, Any]]:
    """Match providers based on user needs using sophisticated vector similarity"""
    try:
        # Get category mapping
        category_map = {
            "plan_protect": "plan_protect",
            "care_aging": "care_aging", 
            "support_death": "support_death"
        }
        
        db_category = category_map.get(category, category)
        providers = PROVIDERS_DB.get(db_category, [])
        
        if not providers:
            return []
        
        # Filter by provider type if specified
        if provider_type:
            providers = [p for p in providers if p["type"] == provider_type]
        
        # Create user needs vector
        user_vector = create_user_needs_vector(user_needs, category)
        
        # Score providers using hybrid approach
        scored_providers = []
        user_words = set(user_needs.lower().split())
        
        for provider in providers:
            # 1. Keyword similarity (Jaccard)
            specialty_words = set(" ".join(provider["specialties"]).lower().split())
            desc_words = set(provider["description"].lower().split())
            provider_words = specialty_words.union(desc_words)
            
            intersection = len(user_words.intersection(provider_words))
            union = len(user_words.union(provider_words))
            keyword_similarity = intersection / union if union > 0 else 0
            
            # 2. Vector similarity (Cosine)
            provider_vector = provider["expertise_vector"]
            vector_similarity = cosine_similarity(user_vector, provider_vector)
            
            # 3. Specialty relevance boost
            specialty_boost = 0
            for word in user_words:
                for specialty in provider["specialties"]:
                    if word in specialty.lower():
                        specialty_boost += 0.1
            specialty_boost = min(0.3, specialty_boost)  # Cap at 0.3
            
            # Combined score: 40% keyword + 50% vector + 10% specialty boost
            final_score = (keyword_similarity * 0.4) + (vector_similarity * 0.5) + specialty_boost
            
            # Create detailed match reasoning
            match_reasons = []
            if keyword_similarity > 0.1:
                match_reasons.append(f"keyword match ({keyword_similarity:.1%})")
            if vector_similarity > 0.7:
                match_reasons.append(f"strong expertise alignment ({vector_similarity:.1%})")
            elif vector_similarity > 0.5:
                match_reasons.append(f"good expertise fit ({vector_similarity:.1%})")
            if specialty_boost > 0:
                match_reasons.append("specialty expertise")
            
            match_reason = f"Match: {', '.join(match_reasons) if match_reasons else 'general fit'}"
            
            scored_providers.append({
                **provider,
                "match_score": final_score,
                "match_reason": match_reason,
                "keyword_similarity": keyword_similarity,
                "vector_similarity": vector_similarity,
                "user_vector": user_vector,  # For debugging
                "provider_vector": provider_vector
            })
        
        # Sort by score and return top 3 (or all if provider_type specified)
        scored_providers.sort(key=lambda x: x["match_score"], reverse=True)
        return scored_providers[:3] if not provider_type else scored_providers
        
    except Exception as e:
        print(f"Matching error: {e}")
        return providers[:3]  # Fallback to first 3

@app.get("/")
async def root():
    return {"message": "Chatbot Demo API"}

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    """Main chat endpoint"""
    try:
        # Build conversation history
        messages = chat_message.conversation_history + [
            {"role": "user", "content": chat_message.message}
        ]
        
        # Get AI response
        ai_response = await get_openai_response(messages, chat_message.category)
        
        # Check if conversation should end (simple heuristic)
        conversation_length = len([msg for msg in messages if msg["role"] == "user"])
        should_complete = conversation_length >= 3 or "summary" in ai_response.lower()
        
        response_data = {
            "response": ai_response,
            "is_complete": should_complete
        }
        
        # If conversation is complete, generate provider types
        if should_complete:
            user_needs = extract_user_needs(messages)
            provider_types = get_provider_types(user_needs, chat_message.category)
            
            response_data["provider_types"] = provider_types
            response_data["user_profile"] = {
                "category": chat_message.category,
                "needs_summary": user_needs[:200] + "..." if len(user_needs) > 200 else user_needs,
                "conversation_length": conversation_length
            }
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/categories")
async def get_categories():
    """Get available service categories"""
    return {
        "categories": [
            {
                "id": "plan_protect",
                "name": "Plan & Protect", 
                "description": "Estate planning, financial planning, and asset protection services",
                "icon": "🛡️"
            },
            {
                "id": "care_aging",
                "name": "Care throughout Aging",
                "description": "Home care, senior living, and aging support services", 
                "icon": "🏠"
            },
            {
                "id": "support_death", 
                "name": "Support upon Death",
                "description": "End-of-life care, estate settlement, and bereavement support",
                "icon": "🕊️"
            }
        ]
    }

@app.get("/providers/{category}")
async def get_providers(category: str):
    """Get providers for a specific category"""
    category_map = {
        "plan_protect": "plan_protect",
        "care_aging": "care_aging",
        "support_death": "support_death"
    }
    
    db_category = category_map.get(category)
    if not db_category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    return {"providers": PROVIDERS_DB[db_category]}

@app.post("/providers/by-type")
async def get_providers_by_type(request: dict):
    """Get specific providers by type based on user needs"""
    try:
        user_needs = request.get("user_needs", "")
        category = request.get("category", "")
        provider_type = request.get("provider_type", "")
        
        if not all([user_needs, category, provider_type]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        recommendations = match_providers(user_needs, category, provider_type)
        
        return {
            "recommendations": recommendations,
            "provider_type": provider_type,
            "category": category
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting providers: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 