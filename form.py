from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, DecimalField


class InputForm(FlaskForm):
    choices = [
        ('Biệt thự', 'Biệt thự'),
        ('Biệt thự liền kề', 'Biệt thự liền kề'),
        ('Chung cư', 'Chung cư'),
        ('Các loại khác', 'Các loại khác'),
        ('Kho xưởng', 'Kho xưởng'),
        ('Mặt bằng', 'Mặt bằng'),
        ('Mặt bằng, cửa hàng', 'Mặt bằng, cửa hàng'),
        ('Nhà hàng, khách sạn', 'Nhà hàng, khách sạn'),
        ('Nhà mặt tiền', 'Nhà mặt tiền'),
        ('Nhà ngõ, hẻm', 'Nhà ngõ, hẻm'),
        ('Nhà phố', 'Nhà phố'),
        ('Nhà riêng', 'Nhà riêng'),
        ('Nhà xưởng', 'Nhà xưởng'),
        ('Nhà đất', 'Nhà đất'),
        ('Phòng trọ, nhà trọ', 'Phòng trọ, nhà trọ'),
        ('Shop, kiot, quán', 'Shop, kiot, quán'),
        ('Trang trại', 'Trang trại'),
        ('Trang trại khu nghỉ dưỡng', 'Trang trại khu nghỉ dưỡng'),
        ('Văn phòng', 'Văn phòng'),
        ('Đất', 'Đất'),
        ('Đất nông, lâm nghiệp', 'Đất nông, lâm nghiệp'),
        ('Đất nền dự án', 'Đất nền dự án'),
        ('Đất nền, phân lô', 'Đất nền, phân lô')
    
    ]
    estate_type = SelectField(choices=choices)
    description = StringField('description')
    square = DecimalField('square')
    submit = SubmitField('Submit')
    