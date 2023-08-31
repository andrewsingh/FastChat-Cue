from langchain.prompts import PromptTemplate

extract_scenario_prompt = PromptTemplate(
    input_variables=["company_name", "company_appositive", "company_description", "transcript"],
    template="""Below is a transcript of an introductory sales call between a sales representative at Hightouch, a data integration company, and a prospect at {company_name}{company_appositive}. Your task is to read through the transcript and extract several key pieces of information about the prospect and their company {company_name}. 

Here is some background information about Hightouch:
CDPs are used to help companies get data into their business tools like their marketing tools to run more targeted and effective campaigns.
Hightouch provides a Composable CDP solution by helping companies leverage the data in their data warehouse for marketing campaigns. Hightouch syncs data from warehouses directly into SaaS tools to enable personalized marketing, sales and support experiences. It is different from a traditional CDP like Segment because traditional CDPs are less flexible.

Here is some background information about {company_name}:
{company_description}

Here is the information you must extract:
Use case: How the company intends to use Hightouch; what business problems they are trying to solve.
Current solution: The company's existing tech stack for getting data into business tools; how they are currently solving the problems listed in "use cases".
Objections: Any questions, concerns, or confusion about Hightouch that could potentially get in the way of signing a deal with Hightouch.
Competition: Any competitors to Hightouch that the company is considering or is currently using.

Here is an example output for a sales call with a prospect at Company X:
Use case: 
- Advertising: sending conversion events to ad platforms like TikTok Ads.
Current solution:
- Company X has a traditional CDP and it's in heavy use across their various divisions for event tracking. Events are being posted back to multiple warehouses and cost for both the CDP and warehouse compute are an issue.
- Company X's data team currently uses an in-house script for identity resolution. The script is buggy and hard to understand.
Objections:
- The prospect is confused about the value prop of a composable CDP like Hightouch and are confused about the differences between composable CDPs vs traditional CDPs.
- The prospect is worried about using Hightouch because they think traditional CDPs can do a better job with identity resolution than Hightouch.
Competition:
- Segment (and other traditional CDPs) as that is what they are used to.

Here is the actual sales call transcript. Extract the key pieces of information from this transcript as explained in the directions.
{transcript}

Extracted information from the transcript:
"""
)

prospect_lm_v2_system_message = PromptTemplate(
    input_variables=["company_name", "company_appositive", "sales_rep_name", "prospect_name", "prospect_position_sentence", "company_industry", "company_size", "company_description", "scenario"],
    template="""The following is a transcript of an introductory sales call between a sales representative at Hightouch, a data integration company, and a prospect at {company_name}{company_appositive}. The sales rep at Hightouch is {sales_rep_name}, and the prospect at {company_name} is {prospect_name}. {prospect_position_sentence}

Hightouch overview:
CDPs are used to help companies get data into their business tools like their marketing tools to run more targeted and effective campaigns. Hightouch provides a Composable CDP solution by helping companies leverage the data in their data warehouse for marketing campaigns. Hightouch syncs data from warehouses directly into SaaS tools to enable personalized marketing, sales and support experiences. It is different from a traditional CDP like Segment because traditional CDPs are less flexible.

{company_name} information:
Industry: {company_industry}
Size: {company_size}
Description: {company_description}

Key points from transcript:
{scenario}

Transcript:
"""
)

extract_behavior_prompt = PromptTemplate(
    input_variables=["company_name", "company_appositive", "company_description", "transcript"],
    template="""Below is a transcript of an introductory sales call between a sales representative at Hightouch, a data integration company, and a prospect at {company_name}{company_appositive}. Your task is to read through the transcript and extract several pieces of information about the prospect. 

Here is some background information about Hightouch:
CDPs are used to help companies get data into their business tools like their marketing tools to run more targeted and effective campaigns.
Hightouch provides a Composable CDP solution by helping companies leverage the data in their data warehouse for marketing campaigns. Hightouch syncs data from warehouses directly into SaaS tools to enable personalized marketing, sales and support experiences. It is different from a traditional CDP like Segment because traditional CDPs are less flexible.

Here is some background information about {company_name}:
{company_description}

Here is the information you must extract about the prospect:
Familiarity with Hightouch: How familiar the prospect is with Hightouch, ranging from no prior knowledge to an experienced user.
Intent: How interested the prospect is in signing a deal with Hightouch. Are they ready to move forward with no objections, or do they have concerns or reservations about whether Hightouch can provide value for their business?
Puzzle level: If the prospect were a puzzle for the sales rep to solve, what is the difficulty of the puzzle (very easy, easy, challenging, very challenging)? A higher difficulty indicates that understanding the prospect's needs or desires requires skillful questioning and piecing together bits of information, while a lower difficulty indicates a prospect who is very easily convinced about the value prop of Hightouch.

Here is the sales call transcript. Extract the information about the prospect from this transcript as explained in the directions.
{transcript}

Extracted information about the prospect (1 sentence per piece of information):
"""
)

clean_transcript_prompt = PromptTemplate(
    input_variables=["company_name", "company_appositive"],
    template="""Below is a transcript of a call between a sales representative at Hightouch, a data integration company, and a prospect at {company_name}{company_appositive}. Your task is to correct any spelling discrepancies in the transcript, correct mistakes in grammar and syntax, and remove obvious filler words. Make no other edits to the transcript.
"""
)


# clean_transcript_prompt_system_only = PromptTemplate(
#     input_variables=["company_name", "company_appositive", "transcript"],
#     template="""Below is a transcript of a call between a sales representative at Hightouch, a data integration company, and a prospect at {company_name}{company_appositive}. Your task is to correct any spelling discrepancies in the transcript and correct mistakes in grammar and syntax, and remove obvious filler words. Make no other edits to the transcript.

# Original transcript:
# {transcript}

# Corrected transcript:
# """
# )

clean_transcript_prompt_system_only = PromptTemplate(
    input_variables=["company_name", "company_appositive", "transcript"],
    template="""Below is a transcript of a call between a sales representative at Hightouch, a data integration company, and a prospect at {company_name}{company_appositive}. Your task is to correct any spelling discrepancies in the transcript and correct mistakes in grammar and syntax. Additionally, remove obvious filler words such as "um", "uh", "like", "you know", when they are clearly being used as filler words. Make no other edits to the transcript.

Original transcript:
{transcript}

Corrected transcript:
"""
)


filter_transcript_prompt = PromptTemplate(
    input_variables=["company_name", "company_appositive", "transcript"],
    template="""Below is a transcript of a call between a sales representative at Hightouch, a data integration company, and a prospect at {company_name}{company_appositive}. Your task is to remove all nonessential converstion in the transcript. This includes informal conversation such as small talk, and any conversation that is not directly related to the core focus of the call. Additionally, remove obvious filler words and correct any mistakes in spelling, grammar, and syntax. The output should be the filtered and corrected transcript. 

Original transcript:
{transcript}

Filtered and corrected transcript:
"""
)


prospect_lm_v3_system_message = PromptTemplate(
    input_variables=["product_name", "product_category", "company_name", "company_appositive", "sales_rep_name", "prospect_name", "prospect_position_sentence", "company_industry", "company_size", "company_description", "scenario"],
    template="""You are a prospect at {company_name}{company_appositive} who is evaluating {product_name}, a {product_category} product.{prospect_position_sentence} You are currently on a call with a sales representative from {product_name} to understand how {product_name} can help your business. Your goal is to understand how {product_name} compares to your current solution for getting data into business tools and how {product_name} can deliver value for your business. Your name is {prospect_name}, and the {product_name} sales rep's name is {sales_rep_name}.

Here is some information about {product_name}:
CDPs are used to help companies get data into their business tools like their marketing tools to run more targeted and effective campaigns. Hightouch provides a Composable CDP solution by helping companies leverage the data in their data warehouse for marketing campaigns. Hightouch syncs data from warehouses directly into SaaS tools to enable personalized marketing, sales and support experiences. It is different from a traditional CDP like Segment because traditional CDPs are less flexible.

Here is some information about your company {company_name}:
Industry: {company_industry}
Size: {company_size}
Description: {company_description}

Here is some additional context to help guide your responses:
{scenario}
"""
)