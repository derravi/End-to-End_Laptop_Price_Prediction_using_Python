from pydantic import BaseModel,Field
from typing import Annotated,Literal

class Userinput(BaseModel):

    Brand : Annotated[str,Field(...,description="Enter the Brand of the Laptop.",examples=['dell'])]
    Model :Annotated[str,Field(...,description="Enter the Model of the laptop.",examples=['Notebook'])]
    CPU : Annotated[str,Field(...,description="Enter the CPU type.",examples=['intel core 8','Ryzen 7'])]
    Status : Annotated[Literal['New','refurbished'],Field(...,description="Enter the Status of the Laptop.",examples=['New','refurbished'])]
    RAM : Annotated[int,Field(...,gt=0,description="Enter the RAM of the laptop.",examples=[8])]
    Storage :Annotated[int,Field(...,gt=0,description="Enter the Storage of the Laptop",examples=[256,512])]
    Storage_type:Annotated[Literal['HDD','SSD'],Field(...,description="Enter the Storage type of laptop.",examples=['HDD','SSD'])]
    GPU :Annotated[str,Field(...,description="Enter the GPU of the laptop.",examples=['RTX 3050'])]
    Screen : Annotated[float,Field(...,description="Enter the Screen size of the laptop.",examples=[15.6])]
    Touch : Annotated[Literal['Yes','No'],Field(...,description="Enter the Touch Screen.",examples=['Yes','No'])]